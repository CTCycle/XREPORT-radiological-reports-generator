from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import delete, exists, func, select
from sqlalchemy.orm import aliased

from XREPORT.server.common.constants import (
    CHECKPOINTS_TABLE,
    CHECKPOINT_EVALUATIONS_TABLE,
    CHECKPOINT_PATH,
    DATASETS_TABLE,
    DATASET_RECORDS_TABLE,
    INFERENCE_REPORTS_TABLE,
    INFERENCE_RUNS_TABLE,
    PROCESSING_RUNS_TABLE,
    TABLE_REQUIRED_COLUMNS,
    TRAINING_SAMPLES_TABLE,
    VALID_IMAGE_EXTENSIONS,
    VALIDATION_IMAGE_STATS_TABLE,
    VALIDATION_PIXEL_DISTRIBUTION_TABLE,
    VALIDATION_RUNS_TABLE,
    VALIDATION_TEXT_SUMMARY_TABLE,
)
from XREPORT.server.common.utils.logger import logger
from XREPORT.server.common.utils.security import validate_checkpoint_name
from XREPORT.server.repositories.database.utils import (
    validate_sql_identifier,
    validate_table_name,
)
from XREPORT.server.repositories.queries.data import DataRepositoryQueries
from XREPORT.server.repositories.schemas import (
    Checkpoint,
    CheckpointEvaluation,
    Dataset,
    DatasetRecord,
    InferenceRun,
    ProcessingRun,
    TrainingSample,
    ValidationImageStat,
    ValidationPixelDistribution,
    ValidationRun,
    ValidationTextSummary,
)

VALID_EXTENSIONS = VALID_IMAGE_EXTENSIONS


###############################################################################
class DataSerializer:
    def __init__(
        self,
        queries: DataRepositoryQueries | None = None,
    ) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.img_shape = (224, 224)
        self.num_channels = 3
        self.valid_extensions = VALID_EXTENSIONS

    # -------------------------------------------------------------------------
    @staticmethod
    def generate_hashcode(metadata: dict[str, Any]) -> str:
        if not metadata:
            return ""

        payload = {
            "dataset_name": metadata.get("dataset_name"),
            "sample_size": metadata.get("sample_size"),
            "validation_size": metadata.get("validation_size"),
            "seed": metadata.get("seed"),
            "vocabulary_size": metadata.get("vocabulary_size"),
            "max_report_size": metadata.get("max_report_size"),
            "tokenizer": metadata.get("tokenizer"),
        }
        serialized = str(sorted(payload.items()))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_json(value: Any, default: Any = None) -> Any:
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            payload = value.strip()
            if not payload:
                return default
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                return default
            if isinstance(decoded, (dict, list)):
                return decoded
            return default
        return default

    # -------------------------------------------------------------------------
    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return (
                value
                if value.tzinfo is not None
                else value.replace(tzinfo=timezone.utc)
            )
        if isinstance(value, str) and value.strip():
            parsed = datetime.fromisoformat(value.strip())
            return (
                parsed
                if parsed.tzinfo is not None
                else parsed.replace(tzinfo=timezone.utc)
            )
        return DataSerializer._now_utc()

    # -------------------------------------------------------------------------
    @staticmethod
    def _format_datetime(value: Any) -> str | None:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, str):
            return value
        return None

    # -------------------------------------------------------------------------
    def validate_required_columns(
        self,
        dataset: pd.DataFrame,
        required_columns: list[str],
        table_name: str,
        operation: str,
    ) -> None:
        missing = [col for col in required_columns if col not in dataset.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {table_name} {operation}: {', '.join(missing)}"
            )

    # -------------------------------------------------------------------------
    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        if offset is not None and offset < 0:
            raise ValueError("offset must be >= 0")
        return self.queries.load_table(table_name, limit=limit, offset=offset)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        return self.queries.count_rows(table_name)

    # -------------------------------------------------------------------------
    def save_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        if dataset.empty:
            logger.debug("Skipping save for %s: dataset is empty", table_name)
            return
        required_columns = TABLE_REQUIRED_COLUMNS.get(table_name)
        if required_columns:
            self.validate_required_columns(
                dataset,
                required_columns,
                table_name,
                "save",
            )
        self.queries.save_table(dataset, table_name)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        if dataset.empty:
            logger.debug("Skipping upsert for %s: dataset is empty", table_name)
            return
        required_columns = TABLE_REQUIRED_COLUMNS.get(table_name)
        if required_columns:
            self.validate_required_columns(
                dataset,
                required_columns,
                table_name,
                "upsert",
            )
        self.queries.upsert_table(dataset, table_name)

    # -------------------------------------------------------------------------
    def _session(self):
        return self.queries.backend.session()

    # -------------------------------------------------------------------------
    def _get_table_class(self, table_name: str):
        safe_table_name = validate_table_name(table_name)
        return self.queries.backend.get_table_class(safe_table_name)

    # -------------------------------------------------------------------------
    def _get_dataset_id(self, dataset_name: str) -> int | None:
        session = self._session()
        try:
            row = session.execute(
                select(Dataset.dataset_id).where(Dataset.name == dataset_name)
            ).first()
        finally:
            session.close()
        return int(row[0]) if row is not None else None

    # -------------------------------------------------------------------------
    def _ensure_dataset(self, dataset_name: str) -> int:
        normalized_name = str(dataset_name or "").strip()
        if not normalized_name:
            raise ValueError("Dataset name cannot be empty")

        existing_id = self._get_dataset_id(normalized_name)
        if existing_id is not None:
            return existing_id

        payload = pd.DataFrame(
            [{"name": normalized_name, "created_at": self._now_utc()}]
        )
        self.upsert_table(payload, DATASETS_TABLE)
        created_id = self._get_dataset_id(normalized_name)
        if created_id is None:
            raise RuntimeError(f"Failed to create dataset: {normalized_name}")
        return created_id

    # -------------------------------------------------------------------------
    def _ensure_checkpoint(self, checkpoint: str) -> int:
        checkpoint_name = validate_checkpoint_name(checkpoint)
        session = self._session()
        try:
            row = session.execute(
                select(Checkpoint.checkpoint_id).where(Checkpoint.name == checkpoint_name)
            ).first()
        finally:
            session.close()
        if row is not None:
            return int(row[0])

        payload = pd.DataFrame(
            [
                {
                    "name": checkpoint_name,
                    "path": os.path.join(CHECKPOINT_PATH, checkpoint_name),
                    "created_at": self._now_utc(),
                }
            ]
        )
        self.upsert_table(payload, CHECKPOINTS_TABLE)
        session = self._session()
        try:
            row = session.execute(
                select(Checkpoint.checkpoint_id).where(Checkpoint.name == checkpoint_name)
            ).first()
        finally:
            session.close()
        if row is None:
            raise RuntimeError(f"Failed to create checkpoint row: {checkpoint_name}")
        return int(row[0])

    # -------------------------------------------------------------------------
    def _delete_by_key(self, table_name: str, column_name: str, value: Any) -> None:
        table_cls = self._get_table_class(table_name)
        safe_column_name = validate_sql_identifier(column_name)
        column = getattr(table_cls, safe_column_name, None)
        if column is None:
            raise ValueError(
                f"Column {safe_column_name} does not exist on table {table_name}"
            )
        session = self._session()
        try:
            session.execute(delete(table_cls).where(column == value))
            session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def validate_metadata(
        self, metadata: dict[str, Any] | Any, target_metadata: dict[str, Any] | Any
    ) -> bool:
        meta_current = dict(metadata or {})
        meta_target = dict(target_metadata or {})
        ignored_keys = {
            "id",
            "date",
            "dataset_id",
            "source_dataset_id",
            "processing_run_id",
        }
        keys_to_compare = [
            key
            for key in set(meta_current) | set(meta_target)
            if key not in ignored_keys
        ]
        differences = {
            key: (meta_current.get(key), meta_target.get(key))
            for key in keys_to_compare
            if meta_current.get(key) != meta_target.get(key)
        }
        return not differences

    # -------------------------------------------------------------------------
    def validate_img_paths(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if "path" not in dataset.columns:
            logger.error(
                "Dataset missing 'path' column - images were not stored with paths"
            )
            return pd.DataFrame()

        valid_mask = dataset["path"].apply(
            lambda item: os.path.isfile(item) if pd.notna(item) else False
        )
        clean_dataset = dataset[valid_mask].reset_index(drop=True)
        dropped = len(dataset) - len(clean_dataset)

        if len(clean_dataset) > 0:
            logger.info("Validated image paths: %s valid records", len(clean_dataset))
        if dropped > 0:
            logger.warning("%s records have missing or invalid image paths", dropped)

        return clean_dataset

    # -------------------------------------------------------------------------
    def get_img_path_from_directory(
        self, path: str, sample_size: float = 1.0
    ) -> list[str]:
        if not os.listdir(path):
            logger.error("No images found in %s, please add them and try again.", path)
            return []

        logger.debug("Valid extensions are: %s", self.valid_extensions)
        images_path: list[str] = []
        for root, _, files in os.walk(path):
            if sample_size < 1.0:
                files = files[: int(sample_size * len(files))]
            for file in files:
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    images_path.append(os.path.join(root, file))
        return images_path

    # -------------------------------------------------------------------------
    def load_source_dataset(
        self,
        sample_size: float = 1.0,
        seed: int = 42,
        dataset_name: str | None = None,
    ) -> pd.DataFrame:
        dataset_name_filter = str(dataset_name).strip() if dataset_name else None
        stmt = (
            select(
                Dataset.dataset_id,
                Dataset.name.label("name"),
                DatasetRecord.record_id,
                DatasetRecord.image_name.label("image"),
                DatasetRecord.report_text.label("text"),
                DatasetRecord.image_path.label("path"),
                DatasetRecord.row_order,
            )
            .join(Dataset, Dataset.dataset_id == DatasetRecord.dataset_id)
            .order_by(
                Dataset.name,
                DatasetRecord.row_order,
                DatasetRecord.record_id,
            )
        )
        if dataset_name_filter:
            stmt = stmt.where(Dataset.name == dataset_name_filter)
        session = self._session()
        try:
            rows = session.execute(stmt).all()
        finally:
            session.close()
        dataset = pd.DataFrame([dict(row._mapping) for row in rows])

        if dataset.empty:
            return dataset
        if sample_size < 1.0:
            dataset = dataset.sample(frac=sample_size, random_state=seed)
        return dataset.reset_index(drop=True)

    # -------------------------------------------------------------------------
    def _load_latest_processing_run(
        self, dataset_name: str | None = None
    ) -> dict[str, Any] | None:
        dataset_name_filter = str(dataset_name).strip() if dataset_name else None
        source_dataset = aliased(Dataset)
        stmt = (
            select(
                ProcessingRun.processing_run_id,
                ProcessingRun.dataset_id,
                ProcessingRun.source_dataset_id,
                ProcessingRun.config_hash,
                ProcessingRun.executed_at,
                ProcessingRun.seed,
                ProcessingRun.sample_size,
                ProcessingRun.validation_size,
                ProcessingRun.split_seed,
                ProcessingRun.vocabulary_size,
                ProcessingRun.max_report_size,
                ProcessingRun.tokenizer,
                Dataset.name.label("dataset_name"),
                source_dataset.name.label("source_dataset"),
            )
            .join(Dataset, Dataset.dataset_id == ProcessingRun.dataset_id)
            .outerjoin(
                source_dataset,
                source_dataset.dataset_id == ProcessingRun.source_dataset_id,
            )
            .order_by(ProcessingRun.processing_run_id.desc())
            .limit(1)
        )
        if dataset_name_filter:
            stmt = stmt.where(Dataset.name == dataset_name_filter)

        session = self._session()
        try:
            row = session.execute(stmt).first()
        finally:
            session.close()
        if row is None:
            return None
        return dict(row._mapping)

    # -------------------------------------------------------------------------
    def load_training_data(
        self,
        only_metadata: bool = False,
        dataset_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | dict[str, Any]:
        latest_run = self._load_latest_processing_run(dataset_name)
        if latest_run is None:
            logger.warning("No processing runs found in database")
            if only_metadata:
                return {}
            return pd.DataFrame(), pd.DataFrame(), {}

        metadata = {
            "name": latest_run["dataset_name"],
            "source_dataset": latest_run.get("source_dataset"),
            "hashcode": latest_run.get("config_hash"),
            "date": self._format_datetime(latest_run.get("executed_at")),
            "seed": latest_run.get("seed"),
            "sample_size": latest_run.get("sample_size"),
            "validation_size": latest_run.get("validation_size"),
            "split_seed": latest_run.get("split_seed"),
            "vocabulary_size": latest_run.get("vocabulary_size"),
            "max_report_size": latest_run.get("max_report_size"),
            "tokenizer": latest_run.get("tokenizer"),
            "processing_run_id": latest_run.get("processing_run_id"),
            "dataset_id": latest_run.get("dataset_id"),
            "source_dataset_id": latest_run.get("source_dataset_id"),
        }
        if only_metadata:
            return metadata

        stmt = (
            select(
                TrainingSample.training_sample_id,
                DatasetRecord.record_id,
                TrainingSample.split,
                TrainingSample.tokens_json.label("tokens"),
                DatasetRecord.image_name.label("image"),
                DatasetRecord.report_text.label("text"),
                DatasetRecord.image_path.label("path"),
            )
            .join(DatasetRecord, DatasetRecord.record_id == TrainingSample.record_id)
            .where(
                TrainingSample.processing_run_id == latest_run["processing_run_id"]
            )
            .order_by(TrainingSample.training_sample_id)
        )
        session = self._session()
        try:
            rows = session.execute(stmt).all()
        finally:
            session.close()
        training_data = pd.DataFrame([dict(row._mapping) for row in rows])

        if training_data.empty:
            return pd.DataFrame(), pd.DataFrame(), metadata

        training_data["tokens"] = training_data["tokens"].apply(
            lambda value: DataSerializer._parse_json(value, default=[])
        )
        train_data = training_data[training_data["split"] == "train"].copy()
        val_data = training_data[training_data["split"] == "validation"].copy()
        return train_data, val_data, metadata

    # -------------------------------------------------------------------------
    def save_training_data(
        self,
        configuration: dict[str, Any],
        training_data: pd.DataFrame,
        vocabulary_size: int | None = None,
        hashcode: str | None = None,
    ) -> None:
        if training_data.empty:
            raise ValueError("Training dataset is empty; nothing to save.")
        if not hashcode:
            raise ValueError("Training dataset hashcode is required.")

        self.validate_required_columns(
            training_data,
            ["record_id", "image", "text", "tokens", "split", "path"],
            TRAINING_SAMPLES_TABLE,
            "prepare",
        )

        dataset_name = str(configuration.get("dataset_name", "")).strip()
        source_dataset = str(configuration.get("source_dataset", "")).strip()
        if not dataset_name:
            raise ValueError("Training configuration must include a dataset_name.")
        if not source_dataset:
            raise ValueError("Training configuration must include a source_dataset.")

        dataset_id = self._ensure_dataset(dataset_name)
        source_dataset_id = self._ensure_dataset(source_dataset)

        run_payload = pd.DataFrame(
            [
                {
                    "dataset_id": dataset_id,
                    "source_dataset_id": source_dataset_id,
                    "config_hash": hashcode,
                    "executed_at": self._now_utc(),
                    "seed": int(configuration.get("seed", 42)),
                    "sample_size": float(configuration.get("sample_size", 1.0)),
                    "validation_size": float(configuration.get("validation_size", 0.2)),
                    "split_seed": int(configuration.get("split_seed", 42)),
                    "vocabulary_size": vocabulary_size,
                    "max_report_size": int(configuration.get("max_report_size", 200)),
                    "tokenizer": str(configuration.get("tokenizer") or ""),
                }
            ]
        )
        self.upsert_table(run_payload, PROCESSING_RUNS_TABLE)
        session = self._session()
        try:
            run_row = session.execute(
                select(ProcessingRun.processing_run_id)
                .where(
                    ProcessingRun.config_hash == hashcode,
                    ProcessingRun.dataset_id == dataset_id,
                )
                .order_by(ProcessingRun.processing_run_id.desc())
                .limit(1)
            ).first()
            if run_row is None:
                raise RuntimeError("Processing run was not persisted correctly")
            processing_run_id = int(run_row[0])
            source_records = session.execute(
                select(DatasetRecord.record_id).where(
                    DatasetRecord.dataset_id == source_dataset_id
                )
            ).all()
        finally:
            session.close()

        if not source_records:
            raise ValueError(f"No source records found for dataset: {source_dataset}")

        training_payload = training_data.copy()
        training_payload["record_id"] = pd.to_numeric(
            training_payload["record_id"],
            errors="coerce",
        )
        missing_record_ids = int(training_payload["record_id"].isna().sum())
        if missing_record_ids:
            raise ValueError(
                f"Training payload has {missing_record_ids} rows without record_id"
            )
        training_payload["record_id"] = training_payload["record_id"].astype(int)

        source_record_ids = {int(row[0]) for row in source_records}
        invalid_record_ids = int(
            (~training_payload["record_id"].isin(source_record_ids)).sum()
        )
        if invalid_record_ids:
            raise ValueError(
                f"Training payload has {invalid_record_ids} rows referencing records outside source dataset"
            )

        normalized_split = (
            training_payload["split"].astype("string").str.strip().str.lower()
        )
        invalid_split_mask = ~normalized_split.isin(["train", "validation"])
        invalid_split_count = int(invalid_split_mask.sum())
        if invalid_split_count:
            raise ValueError(
                f"Training payload has {invalid_split_count} rows with invalid split values"
            )

        samples_df = pd.DataFrame(
            {
                "processing_run_id": processing_run_id,
                "record_id": training_payload["record_id"],
                "split": normalized_split.astype(str),
                "tokens_json": training_payload["tokens"],
            }
        )
        self._delete_by_key(
            TRAINING_SAMPLES_TABLE,
            "processing_run_id",
            processing_run_id,
        )
        self.upsert_table(samples_df, TRAINING_SAMPLES_TABLE)

    # -------------------------------------------------------------------------
    def upsert_source_dataset(self, dataset: pd.DataFrame) -> None:
        self.validate_required_columns(
            dataset,
            ["dataset_name", "image_name", "report_text", "image_path"],
            DATASET_RECORDS_TABLE,
            "prepare",
        )

        dataset_payload = dataset.copy()
        dataset_payload["dataset_name"] = dataset_payload["dataset_name"].astype(str)
        dataset_payload["image_name"] = dataset_payload["image_name"].astype(str)
        dataset_payload["report_text"] = (
            dataset_payload["report_text"].fillna("").astype(str)
        )
        dataset_payload["image_path"] = dataset_payload["image_path"].astype(str)

        batches: list[pd.DataFrame] = []
        for dataset_name, group in dataset_payload.groupby("dataset_name", sort=False):
            dataset_id = self._ensure_dataset(dataset_name)
            records = group.copy().reset_index(drop=True)
            if "row_order" not in records.columns:
                records["row_order"] = range(1, len(records) + 1)

            batches.append(
                pd.DataFrame(
                    {
                        "dataset_id": dataset_id,
                        "image_name": records["image_name"],
                        "report_text": records["report_text"],
                        "image_path": records["image_path"],
                        "row_order": records["row_order"].astype(int),
                    }
                )
            )

        if not batches:
            return
        self.upsert_table(pd.concat(batches, ignore_index=True), DATASET_RECORDS_TABLE)

    # -------------------------------------------------------------------------
    def save_generated_reports(
        self,
        reports: list[dict[str, Any]],
        generation_mode: str = "unknown",
        request_id: str | None = None,
    ) -> None:
        if not reports:
            return
        checkpoint = str(reports[0].get("checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError("Generated reports payload requires checkpoint")

        checkpoint_id = self._ensure_checkpoint(checkpoint)
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            normalized_request_id = f"gen_{uuid.uuid4().hex[:12]}"
        normalized_generation_mode = str(generation_mode or "").strip() or "unknown"

        run_df = pd.DataFrame(
            [
                {
                    "checkpoint_id": checkpoint_id,
                    "generation_mode": normalized_generation_mode,
                    "request_id": normalized_request_id,
                    "executed_at": self._now_utc(),
                }
            ]
        )
        self.upsert_table(run_df, INFERENCE_RUNS_TABLE)
        session = self._session()
        try:
            run_row = session.execute(
                select(InferenceRun.inference_run_id).where(
                    InferenceRun.request_id == normalized_request_id
                )
            ).first()
        finally:
            session.close()
        if run_row is None:
            raise RuntimeError("Inference run creation failed")
        inference_run_id = int(run_row[0])

        reports_df = pd.DataFrame(reports)
        payload = pd.DataFrame(
            {
                "inference_run_id": inference_run_id,
                "input_image_name": reports_df["image"].astype(str),
                "generated_report": reports_df["report"].astype(str),
                "record_id": None,
            }
        )
        self.upsert_table(payload, INFERENCE_REPORTS_TABLE)

    # -------------------------------------------------------------------------
    def save_validation_report(self, report: dict[str, Any]) -> None:
        dataset_name = str(report.get("dataset_name") or "").strip()
        if not dataset_name:
            raise ValueError("Validation report requires dataset_name")

        dataset_id = self._ensure_dataset(dataset_name)
        run_df = pd.DataFrame(
            [
                {
                    "dataset_id": dataset_id,
                    "executed_at": self._coerce_datetime(report.get("date")),
                    "sample_size": float(report.get("sample_size") or 1.0),
                    "metrics_json": report.get("metrics") or [],
                    "artifacts_json": report.get("artifacts") or {},
                }
            ]
        )
        self.upsert_table(run_df, VALIDATION_RUNS_TABLE)
        session = self._session()
        try:
            row = session.execute(
                select(ValidationRun.validation_run_id)
                .where(ValidationRun.dataset_id == dataset_id)
                .order_by(ValidationRun.validation_run_id.desc())
                .limit(1)
            ).first()
        finally:
            session.close()
        if row is None:
            raise RuntimeError("Validation run creation failed")
        validation_run_id = int(row[0])

        text_stats = report.get("text_statistics") or {}
        text_summary_df = pd.DataFrame(
            [
                {
                    "validation_run_id": validation_run_id,
                    "count": int(text_stats.get("count", 0) or 0),
                    "total_words": int(text_stats.get("total_words", 0) or 0),
                    "unique_words": int(text_stats.get("unique_words", 0) or 0),
                    "avg_words_per_report": float(
                        text_stats.get("avg_words_per_report", 0.0) or 0.0
                    ),
                    "min_words_per_report": int(
                        text_stats.get("min_words_per_report", 0) or 0
                    ),
                    "max_words_per_report": int(
                        text_stats.get("max_words_per_report", 0) or 0
                    ),
                }
            ]
        )
        self.upsert_table(text_summary_df, VALIDATION_TEXT_SUMMARY_TABLE)

        self._delete_by_key(
            VALIDATION_IMAGE_STATS_TABLE,
            "validation_run_id",
            validation_run_id,
        )
        image_records = report.get("image_records") or []
        image_df = pd.DataFrame(image_records)
        if not image_df.empty:
            if "record_id" not in image_df.columns:
                logger.warning(
                    "Skipping image statistics persistence because record_id is missing from payload"
                )
            else:
                image_df["record_id"] = pd.to_numeric(
                    image_df["record_id"],
                    errors="coerce",
                )
                invalid_missing = int(image_df["record_id"].isna().sum())
                if invalid_missing:
                    logger.warning(
                        "Skipped %s image statistics rows with invalid record_id",
                        invalid_missing,
                    )
                image_df = image_df[image_df["record_id"].notna()].copy()
                image_df["record_id"] = image_df["record_id"].astype(int)

                session = self._session()
                try:
                    dataset_record_rows = session.execute(
                        select(DatasetRecord.record_id).where(
                            DatasetRecord.dataset_id == dataset_id
                        )
                    ).all()
                finally:
                    session.close()
                valid_record_ids = {int(row[0]) for row in dataset_record_rows}
                invalid_dataset_refs = int(
                    (~image_df["record_id"].isin(valid_record_ids)).sum()
                )
                if invalid_dataset_refs:
                    logger.warning(
                        "Skipped %s image statistics rows not belonging to dataset",
                        invalid_dataset_refs,
                    )
                image_df = image_df[image_df["record_id"].isin(valid_record_ids)].copy()

                if not image_df.empty:
                    payload = pd.DataFrame(
                        {
                            "validation_run_id": validation_run_id,
                            "record_id": image_df["record_id"].astype(int),
                            "height": image_df.get("height"),
                            "width": image_df.get("width"),
                            "mean": image_df.get("mean"),
                            "median": image_df.get("median"),
                            "std": image_df.get("std"),
                            "min": image_df.get("min"),
                            "max": image_df.get("max"),
                            "pixel_range": image_df.get("pixel_range"),
                            "noise_std": image_df.get("noise_std"),
                            "noise_ratio": image_df.get("noise_ratio"),
                        }
                    )
                    self.upsert_table(payload, VALIDATION_IMAGE_STATS_TABLE)

        self._delete_by_key(
            VALIDATION_PIXEL_DISTRIBUTION_TABLE,
            "validation_run_id",
            validation_run_id,
        )
        pixel_distribution = report.get("pixel_distribution") or {}
        bins = pixel_distribution.get("bins") or []
        counts = pixel_distribution.get("counts") or []
        size = min(len(bins), len(counts))
        if size > 0:
            pixel_df = pd.DataFrame(
                {
                    "validation_run_id": [validation_run_id] * size,
                    "bin": [int(value) for value in bins[:size]],
                    "count": [int(value) for value in counts[:size]],
                }
            )
            self.upsert_table(pixel_df, VALIDATION_PIXEL_DISTRIBUTION_TABLE)

    # -------------------------------------------------------------------------
    def get_validation_report(self, dataset_name: str) -> dict[str, Any] | None:
        dataset_id = self._get_dataset_id(str(dataset_name or "").strip())
        if dataset_id is None:
            return None
        session = self._session()
        try:
            run_row = session.execute(
                select(
                    ValidationRun.validation_run_id,
                    ValidationRun.executed_at,
                    ValidationRun.sample_size,
                    ValidationRun.metrics_json,
                    ValidationRun.artifacts_json,
                )
                .where(ValidationRun.dataset_id == dataset_id)
                .order_by(ValidationRun.validation_run_id.desc())
                .limit(1)
            ).first()
            if run_row is None:
                return None

            validation_run_id = int(run_row[0])
            text_row = session.execute(
                select(
                    ValidationTextSummary.count,
                    ValidationTextSummary.total_words,
                    ValidationTextSummary.unique_words,
                    ValidationTextSummary.avg_words_per_report,
                    ValidationTextSummary.min_words_per_report,
                    ValidationTextSummary.max_words_per_report,
                ).where(ValidationTextSummary.validation_run_id == validation_run_id)
            ).first()

            image_agg = session.execute(
                select(
                    func.count(ValidationImageStat.record_id),
                    func.avg(ValidationImageStat.height),
                    func.avg(ValidationImageStat.width),
                    func.avg(ValidationImageStat.mean),
                    func.avg(ValidationImageStat.std),
                    func.avg(ValidationImageStat.noise_std),
                    func.avg(ValidationImageStat.noise_ratio),
                ).where(ValidationImageStat.validation_run_id == validation_run_id)
            ).first()

            pixel_rows = session.execute(
                select(
                    ValidationPixelDistribution.bin,
                    ValidationPixelDistribution.count,
                )
                .where(ValidationPixelDistribution.validation_run_id == validation_run_id)
                .order_by(ValidationPixelDistribution.bin)
            ).all()
        finally:
            session.close()

        metrics = DataSerializer._parse_json(run_row[3], default=[])
        artifacts = DataSerializer._parse_json(run_row[4], default={})

        text_statistics = None
        if text_row is not None:
            text_statistics = {
                "count": int(text_row[0] or 0),
                "total_words": int(text_row[1] or 0),
                "unique_words": int(text_row[2] or 0),
                "avg_words_per_report": float(text_row[3] or 0.0),
                "min_words_per_report": int(text_row[4] or 0),
                "max_words_per_report": int(text_row[5] or 0),
            }

        image_statistics = None
        if image_agg is not None and int(image_agg[0] or 0) > 0:
            image_statistics = {
                "count": int(image_agg[0] or 0),
                "mean_height": float(image_agg[1] or 0.0),
                "mean_width": float(image_agg[2] or 0.0),
                "mean_pixel_value": float(image_agg[3] or 0.0),
                "std_pixel_value": float(image_agg[4] or 0.0),
                "mean_noise_std": float(image_agg[5] or 0.0),
                "mean_noise_ratio": float(image_agg[6] or 0.0),
            }

        pixel_distribution = None
        if pixel_rows:
            pixel_distribution = {
                "bins": [int(row[0]) for row in pixel_rows],
                "counts": [int(row[1]) for row in pixel_rows],
            }

        return {
            "dataset_name": dataset_name,
            "date": self._format_datetime(run_row[1]),
            "sample_size": float(run_row[2] or 0.0),
            "metrics": metrics if isinstance(metrics, list) else [],
            "text_statistics": text_statistics,
            "image_statistics": image_statistics,
            "pixel_distribution": pixel_distribution,
            "artifacts": artifacts if isinstance(artifacts, dict) else {},
        }

    # -------------------------------------------------------------------------
    def validation_report_exists(self, dataset_name: str) -> bool:
        dataset_id = self._get_dataset_id(str(dataset_name or "").strip())
        if dataset_id is None:
            return False
        session = self._session()
        try:
            return bool(
                session.execute(
                    select(exists().where(ValidationRun.dataset_id == dataset_id))
                ).scalar()
            )
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def save_checkpoint_evaluation_report(self, report: dict[str, Any]) -> None:
        checkpoint = str(report.get("checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError("Checkpoint evaluation report requires a checkpoint name")

        checkpoint_id = self._ensure_checkpoint(checkpoint)
        payload = pd.DataFrame(
            [
                {
                    "checkpoint_id": checkpoint_id,
                    "executed_at": self._coerce_datetime(report.get("date")),
                    "metrics_json": report.get("metrics") or [],
                    "metric_configs_json": report.get("metric_configs") or {},
                    "results_json": report.get("results") or {},
                }
            ]
        )
        self.upsert_table(payload, CHECKPOINT_EVALUATIONS_TABLE)

    # -------------------------------------------------------------------------
    def get_checkpoint_evaluation_report(
        self, checkpoint: str
    ) -> dict[str, Any] | None:
        checkpoint_name = str(checkpoint or "").strip()
        if not checkpoint_name:
            return None
        session = self._session()
        try:
            row = session.execute(
                select(
                    CheckpointEvaluation.executed_at,
                    CheckpointEvaluation.metrics_json,
                    CheckpointEvaluation.metric_configs_json,
                    CheckpointEvaluation.results_json,
                )
                .join(
                    Checkpoint,
                    Checkpoint.checkpoint_id == CheckpointEvaluation.checkpoint_id,
                )
                .where(Checkpoint.name == checkpoint_name)
                .order_by(CheckpointEvaluation.evaluation_id.desc())
                .limit(1)
            ).first()
        finally:
            session.close()
        if row is None:
            return None

        metrics = DataSerializer._parse_json(row[1], default=[])
        metric_configs = DataSerializer._parse_json(row[2], default={})
        results = DataSerializer._parse_json(row[3], default={})
        return {
            "checkpoint": checkpoint_name,
            "date": self._format_datetime(row[0]),
            "metrics": metrics if isinstance(metrics, list) else [],
            "metric_configs": metric_configs
            if isinstance(metric_configs, dict)
            else {},
            "results": results if isinstance(results, dict) else {},
        }

    # -------------------------------------------------------------------------
    def checkpoint_evaluation_report_exists(self, checkpoint: str) -> bool:
        checkpoint_name = str(checkpoint or "").strip()
        if not checkpoint_name:
            return False
        session = self._session()
        try:
            return bool(
                session.execute(
                    select(
                        exists().where(
                            CheckpointEvaluation.checkpoint_id
                            == Checkpoint.checkpoint_id,
                            Checkpoint.name == checkpoint_name,
                        )
                    )
                ).scalar()
            )
        finally:
            session.close()

