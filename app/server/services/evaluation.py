from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from keras import Model
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader

from server.common.utils.logger import logger
from server.models.inference import TextGenerator

###############################################################################
class CheckpointInputMismatchError(ValueError):
    """Raised when checkpoint input shapes do not match evaluation data."""

###############################################################################
class CheckpointEvaluator:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        model: Model,
        train_config: dict[str, Any],
        model_metadata: dict[str, Any],
    ) -> None:
        self.model = model
        self.train_config = train_config
        self.model_metadata = model_metadata
        self.max_report_size = model_metadata.get("max_report_size", 200)

    # -------------------------------------------------------------------------
    def evaluate_model(self, validation_dataset: DataLoader) -> dict[str, float]:
        logger.info("Running model evaluation on validation dataset...")
        try:
            validation_results = self.model.evaluate(
                validation_dataset,
                verbose="auto",
            )

            # Model returns [loss, accuracy] for compiled metrics
            loss = float(validation_results[0]) if len(validation_results) > 0 else 0.0
            accuracy = (
                float(validation_results[1]) if len(validation_results) > 1 else 0.0
            )

            logger.info(
                f"Evaluation complete - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
            )

            return {
                "loss": loss,
                "accuracy": accuracy,
            }
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise

    # -------------------------------------------------------------------------
    def preflight_validation_dataset(self, validation_dataset: DataLoader) -> None:
        """Validate and, when necessary, build the model from a real data batch."""
        try:
            batch = next(iter(validation_dataset))
        except StopIteration as exc:
            raise CheckpointInputMismatchError(
                "Checkpoint validation data is empty."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise CheckpointInputMismatchError(
                f"Unable to read a checkpoint validation batch: {exc}"
            ) from exc

        model_inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not isinstance(model_inputs, (tuple, list)):
            model_inputs = (model_inputs,)
        actual_shapes = [
            self._without_batch_dimension(tuple(getattr(value, "shape", ())))
            for value in model_inputs
        ]
        expected_shapes = self._model_input_shapes()

        if expected_shapes and len(expected_shapes) != len(actual_shapes):
            raise CheckpointInputMismatchError(
                "Checkpoint expects "
                f"{len(expected_shapes)} input(s), but validation data provides "
                f"{len(actual_shapes)}."
            )

        if expected_shapes:
            mismatches = [
                (index, expected, actual)
                for index, (expected, actual) in enumerate(
                    zip(expected_shapes, actual_shapes, strict=True)
                )
                if not self._shapes_compatible(expected, actual)
            ]
            if mismatches:
                details = "; ".join(
                    f"input {index}: expected {expected}, found {actual}"
                    for index, expected, actual in mismatches
                )
                raise CheckpointInputMismatchError(
                    f"Checkpoint input shape mismatch ({details})."
                )

        if not getattr(self.model, "built", True):
            try:
                self.model.build(
                    [
                        (None, *shape)
                        for shape in actual_shapes
                    ]
                )
            except Exception as exc:  # noqa: BLE001
                raise CheckpointInputMismatchError(
                    f"Unable to build checkpoint model for validation inputs: {exc}"
                ) from exc

    # -------------------------------------------------------------------------
    def _model_input_shapes(self) -> list[tuple[Any, ...]]:
        raw_shapes = getattr(self.model, "input_shape", None)
        if raw_shapes is None:
            return []
        if (
            isinstance(raw_shapes, tuple)
            and raw_shapes
            and (raw_shapes[0] is None or isinstance(raw_shapes[0], int))
        ):
            raw_shapes = [raw_shapes]
        if not isinstance(raw_shapes, (tuple, list)):
            return []
        return [
            tuple(shape[1:]) if len(shape) > 0 and shape[0] is None else tuple(shape)
            for shape in raw_shapes
            if isinstance(shape, (tuple, list))
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _shapes_compatible(
        expected: tuple[Any, ...],
        actual: tuple[Any, ...],
    ) -> bool:
        return len(expected) == len(actual) and all(
            expected_value is None or int(expected_value) == int(actual_value)
            for expected_value, actual_value in zip(expected, actual, strict=True)
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _without_batch_dimension(shape: tuple[Any, ...]) -> tuple[Any, ...]:
        return shape[1:] if shape else shape

    # -------------------------------------------------------------------------
    def normalize_report_text(
        self,
        value: Any,
        image_path: str,
        label: str,
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized if normalized else None
        if isinstance(value, bytes):
            normalized = value.decode("utf-8", errors="ignore").strip()
            return normalized if normalized else None
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return None
            logger.warning(
                "Skipping %s report for %s: expected string, got float",
                label,
                image_path,
            )
            return None
        if isinstance(value, (int, np.integer)):
            logger.warning(
                "Skipping %s report for %s: expected string, got int",
                label,
                image_path,
            )
            return None
        logger.warning(
            "Skipping %s report for %s: expected string, got %s",
            label,
            image_path,
            type(value).__name__,
        )
        return None

    # -------------------------------------------------------------------------
    def calculate_bleu_score(
        self,
        validation_data: pd.DataFrame,
        num_samples: int,
    ) -> float:
        """
        Calculate BLEU score by generating reports for sample images
        and comparing against ground truth.

        Args:
            validation_data: DataFrame with 'path' and 'text' columns
            num_samples: Number of samples to use for BLEU calculation

        Returns:
            Corpus BLEU score (0.0 to 1.0)
        """
        logger.info(f"Calculating BLEU score using {num_samples} samples...")

        if validation_data.empty:
            logger.warning("No validation data provided for BLEU calculation")
            return 0.0
        if (
            "path" not in validation_data.columns
            or "text" not in validation_data.columns
        ):
            logger.warning("Validation data missing required columns for BLEU scoring")
            return 0.0

        # Initialize text generator
        generator = TextGenerator(self.model, self.model_metadata, self.max_report_size)

        # Sample from validation data
        actual_samples = min(num_samples, len(validation_data))
        samples = validation_data.sample(n=actual_samples, random_state=42)
        sampled_images = samples["path"].to_list()
        true_reports = dict(zip(samples["path"], samples["text"]))

        logger.info(f"Generating reports for {actual_samples} images...")

        # Generate reports using greedy decoding
        generated_reports = generator.generate_radiological_reports(
            sampled_images, method="greedy_search"
        )

        if generated_reports is None:
            logger.warning("Failed to generate reports for BLEU scoring")
            return 0.0

        # Prepare references and hypotheses for BLEU calculation
        references = []
        hypotheses = []

        for image_path in sampled_images:
            if image_path in generated_reports and image_path in true_reports:
                reference = self.normalize_report_text(
                    true_reports[image_path],
                    image_path,
                    "reference",
                )
                candidate = self.normalize_report_text(
                    generated_reports[image_path],
                    image_path,
                    "candidate",
                )
                if reference is None or candidate is None:
                    continue
                ref_tokens = reference.lower().split()
                cand_tokens = candidate.lower().split()
                references.append([ref_tokens])
                hypotheses.append(cand_tokens)

        if not references:
            logger.warning("No matching reports found for BLEU calculation")
            return 0.0

        # Calculate corpus BLEU score
        bleu_score = corpus_bleu(references, hypotheses)
        if not isinstance(bleu_score, Real):
            raise TypeError("BLEU calculation returned a non-numeric result")
        logger.info(f"BLEU score: {bleu_score:.4f}")

        return float(bleu_score)
