from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
from keras import Model, ops
from keras.utils import set_random_seed

from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.services.training.dataloader import XRAYDataLoader
from XREPORT.server.utils.services.training.processing import TokenizerHandler


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
        self.img_shape = (224, 224)
        self.num_channels = 3
        self.generator_methods = {
            "greedy_search": self.generate_with_greedy_search,
            "beam_search": self.generate_with_beam_search,
        }

    # -------------------------------------------------------------------------
    def load_tokenizer_and_configuration(
        self,
    ) -> tuple[Any, dict[str, Any]] | None:
        tokenization = TokenizerHandler(self.configuration)
        if tokenization.tokenizer is None:
            return None

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
                if processed_tokens:
                    processed_tokens[-1] += token[2:]
            else:
                processed_tokens.append(token)

        return " ".join(processed_tokens)

    # -------------------------------------------------------------------------
    def translate_tokens_to_text(
        self,
        index_lookup: dict[int, str],
        sequence: Any | np.ndarray,
        tokenizer_config: dict[str, Any],
    ) -> str:
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

        text_tokens = [
            token for token in token_sequence if token not in special_tokens
        ]
        processed_text = self.merge_tokens(text_tokens)

        return processed_text

    # -------------------------------------------------------------------------
    def generate_with_greedy_search(
        self,
        tokenizer_config: dict[str, Any],
        vocabulary: dict[str, int],
        image_path: str,
        stream_callback: Callable[[str, int, int], None] | None = None,
    ) -> str:
        start_token_idx = tokenizer_config["start_token_idx"]
        end_token = tokenizer_config["end_token"]
        index_lookup = {v: k for k, v in vocabulary.items()}

        logger.info(f"Generating report for image {os.path.basename(image_path)}")
        dataloader = XRAYDataLoader(self.configuration, shuffle=False)
        image = dataloader.processor.load_image(image_path, as_array=True)
        image = dataloader.processor.image_normalization(image)
        image = ops.expand_dims(image, axis=0)

        seq_input = ops.zeros((1, self.max_report_size), dtype="int32")
        seq_input = seq_input.numpy()
        seq_input[0, 0] = start_token_idx

        for i in range(1, self.max_report_size):
            predictions = self.model.predict(
                [image, seq_input], verbose=0
            )
            next_token_idx = int(ops.argmax(predictions[0, i - 1, :], axis=-1))
            next_token = index_lookup.get(next_token_idx, "")

            if next_token == end_token:
                break

            seq_input[0, i] = next_token_idx

            if stream_callback is not None:
                stream_callback(next_token, i, self.max_report_size)

        report = self.translate_tokens_to_text(
            index_lookup, seq_input, tokenizer_config
        )
        logger.info(f"Generated report: {report[:100]}...")

        return report

    # -------------------------------------------------------------------------
    def generate_with_beam_search(
        self,
        tokenizer_config: dict[str, Any],
        vocabulary: dict[str, int],
        image_path: str,
        beam_width: int = 3,
        length_penalty: float = 0.6,
        stream_callback: Callable[[str, int, int], None] | None = None,
    ) -> str:
        start_token_idx = tokenizer_config["start_token_idx"]
        end_token_idx = tokenizer_config["end_token_idx"]
        end_token = tokenizer_config["end_token"]
        index_lookup = {v: k for k, v in vocabulary.items()}

        logger.info(f"Generating report for image {os.path.basename(image_path)}")
        dataloader = XRAYDataLoader(self.configuration, shuffle=False)
        image = dataloader.processor.load_image(image_path, as_array=True)
        image = dataloader.processor.image_normalization(image)
        image = ops.expand_dims(image, axis=0)

        # Initialize beam: (sequence, cumulative_log_prob)
        beams: list[tuple[list[int], float]] = [([start_token_idx], 0.0)]

        for step in range(1, self.max_report_size):
            all_candidates: list[tuple[list[int], float]] = []

            for seq, score in beams:
                if seq[-1] == end_token_idx:
                    all_candidates.append((seq, score))
                    continue

                # Build input matching greedy search format
                seq_input = np.zeros((1, self.max_report_size), dtype=np.int32)
                for j, token in enumerate(seq):
                    seq_input[0, j] = token

                predictions = self.model.predict([image, seq_input], verbose=0)
                # Use same indexing as greedy: predictions at position len(seq)-1
                next_token_logits = predictions[0, len(seq) - 1, :]

                log_probs = np.log(np.clip(next_token_logits, 1e-12, 1.0))
                top_indices = np.argsort(log_probs)[-beam_width:][::-1]

                for token_idx in top_indices:
                    new_seq = seq + [int(token_idx)]
                    new_score = score + log_probs[token_idx]
                    all_candidates.append((new_seq, new_score))

            # Apply length normalization and select top beams
            def normalized_score(candidate: tuple[list[int], float]) -> float:
                seq, score = candidate
                return score / (len(seq) ** length_penalty)

            beams = sorted(all_candidates, key=normalized_score, reverse=True)[
                :beam_width
            ]

            # Stream best beam progress
            if stream_callback is not None:
                best_seq = beams[0][0]
                if len(best_seq) > 1:
                    latest_token = index_lookup.get(best_seq[-1], "")
                    if latest_token != end_token:
                        stream_callback(latest_token, step, self.max_report_size)

            # Early stopping if all beams ended
            if all(seq[-1] == end_token_idx for seq, _ in beams):
                break

        best_seq, _ = beams[0]
        seq_output = np.zeros((1, self.max_report_size), dtype=np.int32)
        for i, token in enumerate(best_seq):
            seq_output[0, i] = token

        report = self.translate_tokens_to_text(
            index_lookup, seq_output, tokenizer_config
        )
        logger.info(f"Generated report: {report[:100]}...")

        return report

    # -------------------------------------------------------------------------
    def generate_radiological_reports(
        self,
        images_path: list[str],
        method: str = "greedy_search",
        stream_callback: Callable[[int, str, int, int], None] | None = None,
    ) -> dict[str, str] | None:
        reports: dict[str, str] = {}
        tokenizers_info = self.load_tokenizer_and_configuration()
        if tokenizers_info is None:
            logger.error("Failed to load tokenizer")
            return None

        tokenizer, tokenizer_config = tokenizers_info
        vocabulary = tokenizer.get_vocab()

        generator_fn = self.generator_methods.get(method)
        if generator_fn is None:
            logger.error(f"Unknown generation method: {method}")
            return None

        for idx, path in enumerate(images_path):
            # Create per-image stream callback wrapper
            def image_stream_callback(
                token: str, step: int, total: int, img_idx: int = idx
            ) -> None:
                if stream_callback is not None:
                    stream_callback(img_idx, token, step, total)

            report = generator_fn(
                tokenizer_config,
                vocabulary,
                path,
                stream_callback=image_stream_callback,
            )
            reports[path] = report

        return reports
