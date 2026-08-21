from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from server.common.utils.logger import logger
from server.models.training.processing import (
    TextSanitizer,
    TokenizerHandler,
    TrainValidationSplit,
)
from server.repositories.serialization.dataset import (
    DatasetIntegrityError,
    DatasetRepository,
)
from server.services.jobs import JobExecutionError, JobManager

###############################################################################
def resolve_processed_dataset_name(
    source_dataset_name: str,
    custom_name: str | None,
) -> str:
    if custom_name and custom_name.strip():
        return custom_name.strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{source_dataset_name}_{timestamp}"

###############################################################################
class DatasetProcessingService:
    """Run dataset sanitization, tokenization, splitting, and persistence."""

    def __init__(
        self,
        repository: DatasetRepository,
        job_manager: JobManager,
    ) -> None:
        self.repository = repository
        self.job_manager = job_manager

    # -------------------------------------------------------------------------
    def _save_processed_dataset(
        self,
        configuration: dict[str, Any],
        training_data: pd.DataFrame,
        dataset_name: str,
        source_dataset_name: str,
        vocabulary_size: int,
    ) -> None:
        metadata_for_hash = {
            "dataset_name": dataset_name,
            "seed": configuration.get("seed", 42),
            "sample_size": configuration.get("sample_size", 1.0),
            "validation_size": configuration.get("validation_size", 0.2),
            "vocabulary_size": vocabulary_size,
            "max_report_size": configuration.get("max_report_size", 200),
            "tokenizer": configuration.get("tokenizer", None),
            "source_dataset": source_dataset_name,
        }
        hashcode = self.repository.generate_hashcode(metadata_for_hash)
        self.repository.save_training_data(
            configuration,
            training_data,
            vocabulary_size,
            hashcode,
        )
        logger.info("Preprocessed data saved to database with hash: %s", hashcode)

    # -------------------------------------------------------------------------
    def run(
        self,
        configuration: dict[str, Any],
        job_id: str,
    ) -> dict[str, Any]:
        """Blocking dataset processing function used by the job runtime."""
        source_dataset_name_raw = configuration.get("dataset_name")
        source_dataset_name = (
            str(source_dataset_name_raw).strip() if source_dataset_name_raw else ""
        )
        if not source_dataset_name:
            raise RuntimeError("Dataset name cannot be empty")
        custom_name_raw = configuration.get("custom_name")
        custom_name = str(custom_name_raw) if custom_name_raw is not None else None

        dataset = self.repository.load_source_dataset(
            sample_size=configuration["sample_size"],
            seed=configuration["seed"],
            dataset_name=source_dataset_name,
        )

        if dataset.empty:
            raise RuntimeError(
                f"No data found for dataset: {source_dataset_name}. "
                "Please load the dataset and try again."
            )

        try:
            dataset = self.repository.validate_img_paths(dataset)
        except DatasetIntegrityError as exc:
            raise JobExecutionError(
                str(exc),
                code="dataset_integrity_failed",
                phase="input_validation",
            ) from exc

        dataset_name = resolve_processed_dataset_name(source_dataset_name, custom_name)
        configuration["dataset_name"] = dataset_name
        configuration["source_dataset"] = source_dataset_name

        sanitizer = TextSanitizer(configuration)
        processed_data = sanitizer.sanitize_text(dataset)
        logger.info("Text sanitization completed")

        self.job_manager.update_progress(job_id, 30.0)
        if self.job_manager.should_stop(job_id):
            return {}

        try:
            tokenization = TokenizerHandler(configuration)
            logger.info("Tokenizing text using %s tokenizer", tokenization.tokenizer_id)
            processed_data = tokenization.tokenize_text_corpus(processed_data)
            vocabulary_size = tokenization.vocabulary_size
            logger.info("Vocabulary size: %s tokens", vocabulary_size)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to tokenize text")
            raise RuntimeError(f"Tokenization failed: {exc}") from exc

        self.job_manager.update_progress(job_id, 60.0)
        if self.job_manager.should_stop(job_id):
            return {}

        splitter = TrainValidationSplit(configuration, processed_data)
        training_data = splitter.split_train_and_validation()
        train_samples = len(training_data[training_data["split"] == "train"])
        validation_samples = len(training_data[training_data["split"] == "validation"])
        logger.info(
            "Split complete: %s train, %s validation samples",
            train_samples,
            validation_samples,
        )

        self.job_manager.update_progress(job_id, 80.0)
        if self.job_manager.should_stop(job_id):
            return {}

        try:
            self._save_processed_dataset(
                configuration,
                training_data,
                dataset_name,
                source_dataset_name,
                vocabulary_size,
            )
        except RuntimeError:
            logger.exception("Database error while saving processed dataset")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to save training data")
            raise RuntimeError(f"Failed to save training data: {exc}") from exc

        self.job_manager.update_progress(job_id, 100.0)
        return {
            "total_samples": len(training_data),
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "vocabulary_size": vocabulary_size,
        }

__all__ = [
    "DatasetProcessingService",
    "resolve_processed_dataset_name",
]
