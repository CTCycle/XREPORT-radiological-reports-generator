from __future__ import annotations
import os
from typing import Any

import numpy as np
from keras import Model, ops
from keras.utils import set_random_seed
import tensorflow as tf
from tqdm import tqdm

from XREPORT.app.client.workers import check_thread_status, update_progress_callback
from XREPORT.app.logger import logger
from XREPORT.app.utils.data.loader import XRAYDataLoader
from XREPORT.app.utils.data.process import TokenizerHandler


# [TOOLKIT TO USE THE PRETRAINED MODEL]
###############################################################################
class TextGenerator:
    def __init__(
        self,
        model: Model,
        configuration: dict[str, Any],
        max_report_size: int = 200,
        seed: int = 42,
    ) -> None:
        set_random_seed(seed)
        self.model = model
        self.configuration = configuration
        self.max_report_size = max_report_size
        # define image and text parameters for inference
        self.img_shape = (224, 224)
        self.num_channels = 3
        # report generation methods
        self.generator_methods = {
            "greedy_search": self.generate_with_greed_search,
            "beam_search": self.generate_with_beam_search,
        }

    # -------------------------------------------------------------------------
    def get_images(self, data: list[str]) -> list[np.ndarray | tf.Tensor]:
        loader = XRAYDataLoader(self.configuration)
        images = [loader.processor.load_image(path, as_array=True) for path in data]
        norm_images = [loader.processor.image_normalization(img) for img in images]

        return norm_images

    # -------------------------------------------------------------------------
    def load_tokenizer_and_configuration(self) -> None | tuple[Any, dict[str, Any]]:
        # Get tokenizer and related configuration
        tokenization = TokenizerHandler(self.configuration)
        if tokenization.tokenizer is None:
            return

        tokenizer = tokenization.tokenizer
        tokenizer_parameters = {
            "vocabulary_size": tokenization.vocabulary_size,
            "start_token": tokenizer.cls_token,
            "end_token": tokenizer.sep_token,
            "pad_token": tokenizer.pad_token_id,
            "start_token_idx": tokenizer.convert_tokens_to_ids(tokenizer.cls_token),
            "end_token_idx": tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
        }

        return tokenizer, tokenizer_parameters

    # -------------------------------------------------------------------------
    def merge_tokens(self, tokens: list[str]) -> str:
        processed_tokens = []
        for token in tokens:
            if token.startswith("##"):
                processed_tokens[-1] += token[2:]
            else:
                processed_tokens.append(token)

        joint_text = " ".join(processed_tokens)

        return joint_text

    # -------------------------------------------------------------------------
    def translate_tokens_to_text(
        self,
        index_lookup: Any,
        sequence: Any | np.ndarray,
        tokenizer_config: dict[Any, Any],
    ) -> str:
        # convert indexes to token using tokenizer vocabulary
        # define special tokens and remove them from generated tokens list
        token_sequence = [
            index_lookup[idx.item()]
            for idx in sequence[0, :]
            if idx.item() in index_lookup and idx != 0
        ]

        special_tokens = [
            tokenizer_config["start_token"],
            tokenizer_config["end_token"],
            tokenizer_config["pad_token"],
        ]

        text_tokens = [token for token in token_sequence if token not in special_tokens]
        processed_text = self.merge_tokens(text_tokens)

        return processed_text

    # -------------------------------------------------------------------------
    def generate_with_greed_search(
        self,
        tokenizer_config: dict[Any, Any],
        vocabulary: dict[Any, Any],
        image_path: str,
    ) -> str:
        # extract vocabulary from the tokenizers
        start_token = tokenizer_config["start_token"]
        end_token = tokenizer_config["end_token"]
        pad_token = tokenizer_config["pad_token"]

        # Convert start and end tokens to their corresponding indices
        start_token_idx = tokenizer_config["start_token_idx"]
        index_lookup = {v: k for k, v in vocabulary.items()}

        logger.info(f"Generating report for image {os.path.basename(image_path)}")
        # load image as array, apply normalization and expand batch dimension
        # to make input compliant to model specifics
        dataloader = XRAYDataLoader(self.configuration)
        image = dataloader.processor.load_image(image_path, as_array=True)
        image = dataloader.processor.image_normalization(image)
        image = ops.expand_dims(image, axis=0)
        # initialize an array with same size of max expected report length
        # set the start token as the first element
        seq_input = ops.zeros((1, self.max_report_size), dtype="int32")
        seq_input[0, 0] = start_token_idx  # type: ignore
        # initialize progress bar for better output formatting
        progress_bar = tqdm(total=self.max_report_size)
        for i in range(1, self.max_report_size):
            # predict the next token based on the truncated sequence (last token removed)
            predictions = self.model.predict([image, seq_input], verbose=0)  # type: ignore
            # apply argmax (greedy search) to identify the most probable token
            next_token_idx = ops.argmax(predictions[0, i - 1, :], axis=-1).item()  # type: ignore
            next_token = index_lookup[next_token_idx]
            # Stop sequence generation if end token is generated
            if next_token == end_token:
                progress_bar.n = progress_bar.total
                progress_bar.last_print_n = progress_bar.total
                progress_bar.update(0)
                break

            seq_input[0, i] = next_token_idx  # type: ignore
            progress_bar.update(1)

        progress_bar.close()
        report = self.translate_tokens_to_text(
            index_lookup, seq_input, tokenizer_config
        )

        logger.info(report)

        return report

    # -------------------------------------------------------------------------
    def generate_with_beam_search(
        self,
        tokenizer_config: dict[Any, Any],
        vocabulary: dict[Any, Any],
        image_path: str,
        beam_width: int = 3,
    ) -> str:
        start_token = tokenizer_config["start_token"]
        end_token = tokenizer_config["end_token"]
        start_token_idx = tokenizer_config["start_token_idx"]
        end_token_idx = tokenizer_config["end_token_idx"]
        index_lookup = {v: k for k, v in vocabulary.items()}

        logger.info(f"Generating report for image {os.path.basename(image_path)}")
        dataloader = XRAYDataLoader(self.configuration)
        image = dataloader.processor.load_image(image_path, as_array=True)
        image = dataloader.processor.image_normalization(image)
        image = ops.expand_dims(image, axis=0)

        # Initialize the beam with a single sequence containing only the start token and a score of 0.0 (log-prob)
        beams = [([start_token_idx], 0.0)]

        # Loop over the maximum report length
        for step in range(1, self.max_report_size):
            new_beams = []
            # Expand each beam in the current list
            for seq, score in beams:
                # If the sequence has already generated the end token, carry it forward unchanged.
                if seq[-1] == end_token_idx:
                    new_beams.append((seq, score))
                    continue

                # Prepare a padded sequence input.
                # We create an array of zeros with shape (1, max_report_size) and fill in the current sequence.
                seq_input = ops.zeros((1, self.max_report_size), dtype="int32")
                for j, token in enumerate(seq):
                    seq_input[0, j] = token  # type: ignore

                # Use only the part of the sequence that has been generated so far.
                # (Following your greedy method, the model expects a truncated sequence, excluding the final slot.)
                current_input = seq_input[:, : len(seq)]
                predictions = self.model.predict([image, current_input], verbose=0)  # type: ignore
                # Get the prediction corresponding to the last token in the sequence.
                # In your greedy search, predictions[0, i-1, :] was used; here len(seq)-1 corresponds to the same position.
                next_token_logits = predictions[0, len(seq) - 1, :]
                # Convert logits/probabilities to log probabilities.
                # We clip to avoid log(0) issues.
                log_probs = np.log(np.clip(next_token_logits, 1e-12, 1.0))
                # Select the top `beam_width` token indices.
                top_indices = np.argsort(log_probs)[-beam_width:][::-1]

                # Create new beams for each candidate token.
                for token_idx in top_indices:
                    new_seq = seq + [int(token_idx)]
                    new_score = score + log_probs[token_idx]
                    new_beams.append((new_seq, new_score))

            # Sort all new beams by their cumulative score (in descending order) and keep the top `beam_width` beams.
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            # If every beam in the list ends with the end token, we can stop early.
            if all(seq[-1] == end_token_idx for seq, _ in beams):
                break

        # Choose the best beam (the one with the highest score)
        best_seq, best_score = beams[0]
        # Create a full padded sequence from the best beam for conversion to text.
        seq_input = ops.zeros((1, self.max_report_size), dtype="int32")
        for i, token in enumerate(best_seq):
            seq_input[0, i] = token  # type: ignore

        report = self.translate_tokens_to_text(
            index_lookup, seq_input, tokenizer_config
        )
        logger.info(report)

        return report

    # -------------------------------------------------------------------------
    def generate_radiological_reports(
        self, images_path: list[str], method: str = "greedy_search", **kwargs
    ) -> dict[str, Any] | None:
        reports = {}
        tokenizers_info = self.load_tokenizer_and_configuration()
        if tokenizers_info is None:
            return

        tokenizer, tokenizer_config = tokenizers_info
        vocabulary = tokenizer.get_vocab()
        for i, path in enumerate(images_path):
            report = self.generator_methods[method](tokenizer_config, vocabulary, path)
            reports[path] = report

            # check for thread status and progress bar update
            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(images_path), kwargs.get("progress_callback", None)
            )

        return reports
