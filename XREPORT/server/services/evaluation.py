from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from keras import Model
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader

from XREPORT.server.utils.logger import logger
from XREPORT.server.learning.inference import TextGenerator


###############################################################################
class CheckpointEvaluator:
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
        num_samples: int = 10,
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
        logger.info(f"BLEU score: {bleu_score:.4f}")

        return float(bleu_score)
