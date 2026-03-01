from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from keras import Model, ops
from keras.utils import set_random_seed

from XREPORT.server.common.utils.logger import logger
from XREPORT.server.learning.training.dataloader import XRAYDataLoader
from XREPORT.server.services.processing import TokenizerHandler


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
        self.generator_image_methods = {
            "greedy_search": self.generate_with_greedy_search_from_image,
            "beam_search": self.generate_with_beam_search_from_image,
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
            index_lookup[int(idx)]
            for idx in sequence[0, :]
            if int(idx) in index_lookup and idx != 0
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
    def _build_sequence_input(self, sequence: list[int]) -> np.ndarray:
        seq_input = np.zeros((1, self.max_report_size), dtype=np.int32)
        for index, token in enumerate(sequence):
            seq_input[0, index] = token

        return seq_input

    # -------------------------------------------------------------------------
    def _predict_token_logits(
        self,
        image: np.ndarray,
        seq_input: np.ndarray,
        token_position: int,
    ) -> np.ndarray:
        predictions = self.model([image, seq_input], training=False)
        token_logits = ops.convert_to_numpy(predictions[0, token_position, :])
        return np.asarray(token_logits)

    # -------------------------------------------------------------------------
    def _expand_beam_candidates(
        self,
        image: np.ndarray,
        sequence: list[int],
        score: float,
        beam_width: int,
        end_token_idx: int,
    ) -> list[tuple[list[int], float]]:
        if sequence[-1] == end_token_idx:
            return [(sequence, score)]

        seq_input = self._build_sequence_input(sequence)
        next_token_logits = self._predict_token_logits(
            image,
            seq_input,
            token_position=len(sequence) - 1,
        )

        log_probs = np.log(np.clip(next_token_logits, 1e-12, 1.0))
        top_indices = np.argsort(log_probs)[-beam_width:][::-1]

        return [
            (sequence + [int(token_idx)], score + float(log_probs[token_idx]))
            for token_idx in top_indices
        ]

    # -------------------------------------------------------------------------
    def _select_top_beams(
        self,
        all_candidates: list[tuple[list[int], float]],
        beam_width: int,
        length_penalty: float,
    ) -> list[tuple[list[int], float]]:
        def normalized_score(candidate: tuple[list[int], float]) -> float:
            seq, cumulative_score = candidate
            return cumulative_score / (len(seq) ** length_penalty)

        return sorted(all_candidates, key=normalized_score, reverse=True)[:beam_width]

    # -------------------------------------------------------------------------
    def _stream_best_beam_token(
        self,
        beams: list[tuple[list[int], float]],
        index_lookup: dict[int, str],
        end_token: str,
        step: int,
        stream_callback: Callable[[str, int, int], None] | None,
    ) -> None:
        if stream_callback is None or not beams:
            return

        best_seq = beams[0][0]
        if len(best_seq) <= 1:
            return

        latest_token = index_lookup.get(best_seq[-1], "")
        if latest_token == end_token:
            return

        stream_callback(latest_token, step, self.max_report_size)

    # -------------------------------------------------------------------------
    def _all_beams_ended(
        self,
        beams: list[tuple[list[int], float]],
        end_token_idx: int,
    ) -> bool:
        return all(seq[-1] == end_token_idx for seq, _ in beams)

    # -------------------------------------------------------------------------
    def generate_with_greedy_search_from_image(
        self,
        tokenizer_config: dict[str, Any],
        vocabulary: dict[str, int],
        image: np.ndarray,
        stream_callback: Callable[[str, int, int], None] | None = None,
    ) -> str:
        start_token_idx = tokenizer_config["start_token_idx"]
        end_token = tokenizer_config["end_token"]
        index_lookup = {v: k for k, v in vocabulary.items()}

        # Use numpy for expand_dims to keep data on CPU
        image = np.expand_dims(image, axis=0)

        # Create sequence input as numpy array (stays on CPU)
        seq_input = np.zeros((1, self.max_report_size), dtype=np.int32)
        seq_input[0, 0] = start_token_idx

        for i in range(1, self.max_report_size):
            predictions = self.model([image, seq_input], training=False)
            # Convert predictions to numpy to handle CUDA tensors
            pred_numpy = ops.convert_to_numpy(predictions[0, i - 1, :])
            next_token_idx = int(np.argmax(pred_numpy))
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
    def generate_with_greedy_search(
        self,
        tokenizer_config: dict[str, Any],
        vocabulary: dict[str, int],
        image_path: str,
        stream_callback: Callable[[str, int, int], None] | None = None,
    ) -> str:
        logger.info(f"Generating report for image {os.path.basename(image_path)}")
        dataloader = XRAYDataLoader(self.configuration, shuffle=False)
        image = dataloader.prepare_inference_image(image_path)

        return self.generate_with_greedy_search_from_image(
            tokenizer_config,
            vocabulary,
            image,
            stream_callback=stream_callback,
        )

    # -------------------------------------------------------------------------
    def generate_with_beam_search_from_image(
        self,
        tokenizer_config: dict[str, Any],
        vocabulary: dict[str, int],
        image: np.ndarray,
        beam_width: int = 3,
        length_penalty: float = 0.6,
        stream_callback: Callable[[str, int, int], None] | None = None,
    ) -> str:
        start_token_idx = tokenizer_config["start_token_idx"]
        end_token_idx = tokenizer_config["end_token_idx"]
        end_token = tokenizer_config["end_token"]
        index_lookup = {v: k for k, v in vocabulary.items()}

        # Use numpy for expand_dims to keep data on CPU
        image = np.expand_dims(image, axis=0)

        # Initialize beam: (sequence, cumulative_log_prob)
        beams: list[tuple[list[int], float]] = [([start_token_idx], 0.0)]

        for step in range(1, self.max_report_size):
            all_candidates: list[tuple[list[int], float]] = []

            for seq, score in beams:
                all_candidates.extend(
                    self._expand_beam_candidates(
                        image=image,
                        sequence=seq,
                        score=score,
                        beam_width=beam_width,
                        end_token_idx=end_token_idx,
                    )
                )

            beams = self._select_top_beams(all_candidates, beam_width, length_penalty)
            self._stream_best_beam_token(
                beams=beams,
                index_lookup=index_lookup,
                end_token=end_token,
                step=step,
                stream_callback=stream_callback,
            )

            if self._all_beams_ended(beams, end_token_idx):
                break

        best_seq, _ = beams[0]
        seq_output = self._build_sequence_input(best_seq)

        report = self.translate_tokens_to_text(
            index_lookup, seq_output, tokenizer_config
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
        logger.info(f"Generating report for image {os.path.basename(image_path)}")
        dataloader = XRAYDataLoader(self.configuration, shuffle=False)
        image = dataloader.prepare_inference_image(image_path)

        return self.generate_with_beam_search_from_image(
            tokenizer_config,
            vocabulary,
            image,
            beam_width=beam_width,
            length_penalty=length_penalty,
            stream_callback=stream_callback,
        )

    # -------------------------------------------------------------------------
    def _resolve_generation_methods(
        self,
        method: str,
    ) -> tuple[Callable[..., str], Callable[..., str]] | None:
        generator_fn = self.generator_methods.get(method)
        if generator_fn is None:
            logger.error(f"Unknown generation method: {method}")
            return None

        image_generator_fn = self.generator_image_methods.get(method)
        if image_generator_fn is None:
            logger.error(f"Unknown generation method: {method}")
            return None

        return generator_fn, image_generator_fn

    # -------------------------------------------------------------------------
    def _ensure_numpy_batch(self, batch: Any) -> np.ndarray:
        if isinstance(batch, np.ndarray):
            return batch

        return np.asarray(batch.detach().cpu().numpy())

    # -------------------------------------------------------------------------
    def _generate_reports_from_batches(
        self,
        images_path: list[str],
        tokenizer_config: dict[str, Any],
        vocabulary: dict[str, int],
        image_generator_fn: Callable[..., str],
    ) -> dict[str, str]:
        reports: dict[str, str] = {}
        data = pd.DataFrame({"path": images_path})
        dataloader = XRAYDataLoader(self.configuration, shuffle=False)
        inference_loader = dataloader.build_inference_dataloader(data)

        image_index = 0
        total_images = len(images_path)

        for batch in inference_loader:
            if image_index >= total_images:
                break

            numpy_batch = self._ensure_numpy_batch(batch)
            for item in numpy_batch:
                if image_index >= total_images:
                    break

                report = image_generator_fn(
                    tokenizer_config,
                    vocabulary,
                    item,
                    stream_callback=None,
                )
                reports[images_path[image_index]] = report
                image_index += 1

        return reports

    # -------------------------------------------------------------------------
    def _build_image_stream_callback(
        self,
        stream_callback: Callable[[int, str, int, int], None] | None,
        image_index: int,
    ) -> Callable[[str, int, int], None]:
        def image_stream_callback(token: str, step: int, total: int) -> None:
            if stream_callback is not None:
                stream_callback(image_index, token, step, total)

        return image_stream_callback

    # -------------------------------------------------------------------------
    def _generate_reports_with_streaming(
        self,
        images_path: list[str],
        tokenizer_config: dict[str, Any],
        vocabulary: dict[str, int],
        generator_fn: Callable[..., str],
        stream_callback: Callable[[int, str, int, int], None] | None,
    ) -> dict[str, str]:
        reports: dict[str, str] = {}

        for idx, path in enumerate(images_path):
            report = generator_fn(
                tokenizer_config,
                vocabulary,
                path,
                stream_callback=self._build_image_stream_callback(stream_callback, idx),
            )
            reports[path] = report

        return reports

    # -------------------------------------------------------------------------
    def generate_radiological_reports(
        self,
        images_path: list[str],
        method: str = "greedy_search",
        stream_callback: Callable[[int, str, int, int], None] | None = None,
    ) -> dict[str, str] | None:
        tokenizers_info = self.load_tokenizer_and_configuration()
        if tokenizers_info is None:
            logger.error("Failed to load tokenizer")
            return None

        tokenizer, tokenizer_config = tokenizers_info
        vocabulary = tokenizer.get_vocab()

        generation_methods = self._resolve_generation_methods(method)
        if generation_methods is None:
            return None
        generator_fn, image_generator_fn = generation_methods

        if stream_callback is None and len(images_path) > 1:
            return self._generate_reports_from_batches(
                images_path=images_path,
                tokenizer_config=tokenizer_config,
                vocabulary=vocabulary,
                image_generator_fn=image_generator_fn,
            )

        return self._generate_reports_with_streaming(
            images_path=images_path,
            tokenizer_config=tokenizer_config,
            vocabulary=vocabulary,
            generator_fn=generator_fn,
            stream_callback=stream_callback,
        )
