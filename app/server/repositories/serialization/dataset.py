from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete, exists, func, select
from sqlalchemy.orm import aliased

from server.common.constants import (
    CHECKPOINTS_TABLE,
    CHECKPOINT_EVALUATIONS_TABLE,
    DATASETS_TABLE,
    DATASET_RECORDS_TABLE,
    TABLE_REQUIRED_COLUMNS,
    TRAINING_SAMPLES_TABLE,
    VALID_IMAGE_EXTENSIONS,
)
from server.common.path import CHECKPOINTS_DIR
from server.common.utils.logger import logger
from server.common.utils.security import validate_checkpoint_name
from server.repositories.database.utils import (
    validate_sql_identifier,
    validate_table_name,
)
from server.repositories.database import Database, get_database
from server.repositories.schemas import (
    Checkpoint,
    CheckpointEvaluation,
    Dataset,
    DatasetRecord,
    DatasetVersion,
    ProcessingRun,
    TrainingSample,
    ValidationRun,
)
from server.repositories.schemas.normalization import normalize_key
from server.repositories.serialization.support import JsonDataSupport

VALID_EXTENSIONS = VALID_IMAGE_EXTENSIONS

###############################################################################
class DatasetRepository(JsonDataSupport):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        database: Database | None = None,
    ) -> None:
        self.database = database or get_database()
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
        return JsonDataSupport.parse_json(value, default)

    # -------------------------------------------------------------------------
    @staticmethod
    def _now_utc() -> datetime:
        return JsonDataSupport.now_utc()

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
        return DatasetRepository._now_utc()

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
        return self.database.load_from_database(table_name, limit=limit, offset=offset)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        return self.database.count_rows(table_name)

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
        self.database.save_into_database(dataset, table_name)

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
        self.database.upsert_into_database(dataset, table_name)

    # -------------------------------------------------------------------------
    def _session(self):
        return self.database.session()

    # -------------------------------------------------------------------------
    def _get_table_class(self, table_name: str):
        safe_table_name = validate_table_name(table_name)
        return self.database.get_table_class(safe_table_name)

    # -------------------------------------------------------------------------
    def _get_dataset_id(self, dataset_name: str) -> int | None:
        session = self._session()
        try:
            row = session.execute(
                select(Dataset.dataset_id).where(Dataset.name_key == normalize_key(dataset_name))
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
            [
                {
                    "name": normalized_name,
                    "name_key": normalize_key(normalized_name),
                    "created_at": self._now_utc(),
                }
            ]
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
                select(Checkpoint.checkpoint_id).where(
                    Checkpoint.name_key == normalize_key(checkpoint_name)
                )
            ).first()
        finally:
            session.close()
        if row is not None:
            return int(row[0])

        payload = pd.DataFrame(
            [
                {
                    "name": checkpoint_name,
                    "name_key": normalize_key(checkpoint_name),
                    "path": str(CHECKPOINTS_DIR / checkpoint_name),
                    "created_at": self._now_utc(),
                    "last_seen_at": self._now_utc(),
                }
            ]
        )
        self.upsert_table(payload, CHECKPOINTS_TABLE)
        session = self._session()
        try:
            row = session.execute(
                select(Checkpoint.checkpoint_id).where(
                    Checkpoint.name_key == normalize_key(checkpoint_name)
                )
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
            lambda item: Path(item).is_file() if pd.notna(item) else False
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
        self, path: str | Path, sample_size: float = 1.0
    ) -> list[str]:
        directory_path = Path(path)
        if not any(directory_path.iterdir()):
            logger.error("No images found in %s, please add them and try again.", path)
            return []

        logger.debug("Valid extensions are: %s", self.valid_extensions)
        matching_files = [
            file_path
            for file_path in directory_path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in self.valid_extensions
        ]
        if sample_size < 1.0:
            matching_files = matching_files[: int(sample_size * len(matching_files))]
        return [str(file_path) for file_path in matching_files]

    # -------------------------------------------------------------------------
    def load_source_dataset(
        self,
        sample_size: float = 1.0,
        seed: int = 42,
        dataset_name: str | None = None,
    ) -> pd.DataFrame:
        dataset_name_filter = str(dataset_name).strip() if dataset_name else None
        latest_versions = (
            select(
                DatasetVersion.dataset_id,
                func.max(DatasetVersion.version_number).label("version_number"),
            )
            .group_by(DatasetVersion.dataset_id)
            .subquery()
        )
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
            .join(
                DatasetVersion,
                DatasetVersion.dataset_version_id == DatasetRecord.dataset_version_id,
            )
            .join(
                latest_versions,
                (latest_versions.c.dataset_id == DatasetVersion.dataset_id)
                & (latest_versions.c.version_number == DatasetVersion.version_number),
            )
            .order_by(
                Dataset.name,
                DatasetRecord.row_order,
                DatasetRecord.record_id,
            )
        )
        if dataset_name_filter:
            stmt = stmt.where(Dataset.name_key == normalize_key(dataset_name_filter))
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
            stmt = stmt.where(Dataset.name_key == normalize_key(dataset_name_filter))

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
            lambda value: DatasetRepository._parse_json(value, default=[])
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

        normalized_split = (
            training_payload["split"].astype("string").str.strip().str.lower()
        )
        invalid_split_mask = ~normalized_split.isin(["train", "validation"])
        invalid_split_count = int(invalid_split_mask.sum())
        if invalid_split_count:
            raise ValueError(
                f"Training payload has {invalid_split_count} rows with invalid split values"
            )

        backend = self.database
        with backend.transaction() as session:
            datasets: dict[str, Dataset] = {}
            for name in (dataset_name, source_dataset):
                key = normalize_key(name)
                row = session.execute(
                    select(Dataset).where(Dataset.name_key == key)
                ).scalar_one_or_none()
                if row is None:
                    row = Dataset(name=name, name_key=key, created_at=self._now_utc())
                    session.add(row)
                    session.flush()
                datasets[key] = row
            dataset_id = datasets[normalize_key(dataset_name)].dataset_id
            source_dataset_id = datasets[normalize_key(source_dataset)].dataset_id
            latest_source_version = session.execute(
                select(DatasetVersion.dataset_version_id)
                .where(DatasetVersion.dataset_id == source_dataset_id)
                .order_by(DatasetVersion.version_number.desc())
                .limit(1)
            ).scalar_one_or_none()
            source_records = session.execute(
                select(DatasetRecord.record_id).where(
                    DatasetRecord.dataset_id == source_dataset_id,
                    DatasetRecord.dataset_version_id == latest_source_version,
                )
            ).all()
            if not source_records:
                raise ValueError(f"No source records found for dataset: {source_dataset}")
            source_record_ids = {int(row[0]) for row in source_records}
            invalid_record_ids = int(
                (~training_payload["record_id"].isin(source_record_ids)).sum()
            )
            if invalid_record_ids:
                raise ValueError(
                    f"Training payload has {invalid_record_ids} rows referencing records outside source dataset"
                )
            run = ProcessingRun(
                dataset_id=dataset_id,
                source_dataset_id=source_dataset_id,
                config_hash=hashcode,
                executed_at=self._now_utc(),
                seed=int(configuration.get("seed", 42)),
                sample_size=float(configuration.get("sample_size", 1.0)),
                validation_size=float(configuration.get("validation_size", 0.2)),
                split_seed=int(configuration.get("split_seed", 42)),
                vocabulary_size=vocabulary_size,
                max_report_size=int(configuration.get("max_report_size", 200)),
                tokenizer=str(configuration.get("tokenizer") or ""),
            )
            session.add(run)
            session.flush()
            session.add_all(
                TrainingSample(
                    processing_run_id=run.processing_run_id,
                    record_id=int(row.record_id),
                    split=str(row.split),
                    tokens_json=row.tokens,
                )
                for row in training_payload.assign(split=normalized_split).itertuples(
                    index=False
                )
            )

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

        backend = self.database
        with backend.transaction() as session:
            for dataset_name, group in dataset_payload.groupby("dataset_name", sort=False):
                normalized_name = str(dataset_name).strip()
                dataset_row = session.execute(
                    select(Dataset).where(
                        Dataset.name_key == normalize_key(normalized_name)
                    )
                ).scalar_one_or_none()
                if dataset_row is None:
                    dataset_row = Dataset(
                        name=normalized_name,
                        name_key=normalize_key(normalized_name),
                        created_at=self._now_utc(),
                    )
                    session.add(dataset_row)
                    session.flush()

                records = group.copy().reset_index(drop=True)
                if "row_order" not in records.columns:
                    records["row_order"] = range(1, len(records) + 1)
                canonical_records = [
                    {
                        "image_name": str(row.image_name),
                        "image_name_key": normalize_key(str(row.image_name)),
                        "report_text": str(row.report_text),
                        "image_path": str(row.image_path),
                        "row_order": int(row.row_order),
                    }
                    for row in records.itertuples(index=False)
                ]
                content_hash = hashlib.sha256(
                    json.dumps(
                        canonical_records,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                existing_version = session.execute(
                    select(DatasetVersion).where(
                        DatasetVersion.dataset_id == dataset_row.dataset_id,
                        DatasetVersion.content_hash == content_hash,
                    )
                ).scalar_one_or_none()
                if existing_version is not None:
                    continue
                latest_number = session.execute(
                    select(func.max(DatasetVersion.version_number)).where(
                        DatasetVersion.dataset_id == dataset_row.dataset_id
                    )
                ).scalar() or 0
                version = DatasetVersion(
                    dataset_id=dataset_row.dataset_id,
                    version_number=int(latest_number) + 1,
                    content_hash=content_hash,
                    record_count=len(canonical_records),
                    imported_at=self._now_utc(),
                )
                session.add(version)
                session.flush()
                session.add_all(
                    DatasetRecord(
                        dataset_id=dataset_row.dataset_id,
                        dataset_version_id=version.dataset_version_id,
                        **record,
                    )
                    for record in canonical_records
                )

    # -------------------------------------------------------------------------
    def save_validation_report(self, report: dict[str, Any]) -> None:
        dataset_name = str(report.get("dataset_name") or "").strip()
        if not dataset_name:
            raise ValueError("Validation report requires dataset_name")
        backend = self.database
        with backend.transaction() as session:
            dataset = session.execute(
                select(Dataset).where(Dataset.name_key == normalize_key(dataset_name))
            ).scalar_one_or_none()
            if dataset is None:
                dataset = Dataset(
                    name=dataset_name,
                    name_key=normalize_key(dataset_name),
                    created_at=self._now_utc(),
                )
                session.add(dataset)
                session.flush()
            text_stats = report.get("text_statistics") or {}
            image_stats = report.get("image_statistics") or {}
            pixel_distribution = report.get("pixel_distribution") or {}
            validation_run = ValidationRun(
                dataset_id=dataset.dataset_id,
                executed_at=self._coerce_datetime(report.get("date")),
                sample_size=float(report.get("sample_size") or 1.0),
                metrics_json=report.get("metrics") or [],
                artifacts_json=report.get("artifacts") or {},
                status="succeeded",
                text_count=int(text_stats.get("count", 0) or 0),
                text_total_words=int(text_stats.get("total_words", 0) or 0),
                text_unique_words=int(text_stats.get("unique_words", 0) or 0),
                text_avg_words=float(text_stats.get("avg_words_per_report", 0.0) or 0.0),
                text_min_words=int(text_stats.get("min_words_per_report", 0) or 0),
                text_max_words=int(text_stats.get("max_words_per_report", 0) or 0),
                image_count=int(image_stats.get("count", 0) or 0),
                image_mean_height=float(image_stats.get("mean_height", 0.0) or 0.0),
                image_mean_width=float(image_stats.get("mean_width", 0.0) or 0.0),
                image_mean_value=float(image_stats.get("mean_pixel_value", 0.0) or 0.0),
                image_std_value=float(image_stats.get("std_pixel_value", 0.0) or 0.0),
                image_mean_noise_std=float(image_stats.get("mean_noise_std", 0.0) or 0.0),
                image_mean_noise_ratio=float(image_stats.get("mean_noise_ratio", 0.0) or 0.0),
                pixel_bins_json=pixel_distribution.get("bins") or [],
                pixel_counts_json=pixel_distribution.get("counts") or [],
            )
            session.add(validation_run)
            session.flush()

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
                    ValidationRun.text_count,
                    ValidationRun.text_total_words,
                    ValidationRun.text_unique_words,
                    ValidationRun.text_avg_words,
                    ValidationRun.text_min_words,
                    ValidationRun.text_max_words,
                    ValidationRun.image_count,
                    ValidationRun.image_mean_height,
                    ValidationRun.image_mean_width,
                    ValidationRun.image_mean_value,
                    ValidationRun.image_std_value,
                    ValidationRun.image_mean_noise_std,
                    ValidationRun.image_mean_noise_ratio,
                    ValidationRun.pixel_bins_json,
                    ValidationRun.pixel_counts_json,
                )
                .where(ValidationRun.dataset_id == dataset_id)
                .order_by(ValidationRun.validation_run_id.desc())
                .limit(1)
            ).first()
            if run_row is None:
                return None
        finally:
            session.close()

        metrics = DatasetRepository._parse_json(run_row[3], default=[])
        artifacts = DatasetRepository._parse_json(run_row[4], default={})

        text_statistics = None
        if run_row[5] is not None:
            text_values = run_row[5:11]
            text_statistics = {
                "count": int(text_values[0] or 0),
                "total_words": int(text_values[1] or 0),
                "unique_words": int(text_values[2] or 0),
                "avg_words_per_report": float(text_values[3] or 0.0),
                "min_words_per_report": int(text_values[4] or 0),
                "max_words_per_report": int(text_values[5] or 0),
            }

        image_statistics = None
        image_count = run_row[11] or 0
        if image_count is not None and int(image_count or 0) > 0:
            image_values = run_row[11:18]
            image_statistics = {
                "count": int(image_values[0] or 0),
                "mean_height": float(image_values[1] or 0.0),
                "mean_width": float(image_values[2] or 0.0),
                "mean_pixel_value": float(image_values[3] or 0.0),
                "std_pixel_value": float(image_values[4] or 0.0),
                "mean_noise_std": float(image_values[5] or 0.0),
                "mean_noise_ratio": float(image_values[6] or 0.0),
            }

        pixel_distribution = None
        pixel_bins = DatasetRepository._parse_json(run_row[18], default=[])
        pixel_counts = DatasetRepository._parse_json(run_row[19], default=[])
        if pixel_bins and pixel_counts:
            pixel_distribution = {
                "bins": [int(value) for value in pixel_bins],
                "counts": [int(value) for value in pixel_counts],
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
                .where(Checkpoint.name_key == normalize_key(checkpoint_name))
                .order_by(CheckpointEvaluation.evaluation_id.desc())
                .limit(1)
            ).first()
        finally:
            session.close()
        if row is None:
            return None

        metrics = DatasetRepository._parse_json(row[1], default=[])
        metric_configs = DatasetRepository._parse_json(row[2], default={})
        results = DatasetRepository._parse_json(row[3], default={})
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
                            Checkpoint.name_key == normalize_key(checkpoint_name),
                        )
                    )
                ).scalar()
            )
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def list_checkpoint_evaluations(
        self, checkpoint: str | None = None, *, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        stmt = (
            select(
                CheckpointEvaluation.request_id,
                CheckpointEvaluation.status,
                CheckpointEvaluation.executed_at,
                CheckpointEvaluation.metrics_json,
                CheckpointEvaluation.results_json,
                Checkpoint.name,
            )
            .join(Checkpoint, Checkpoint.checkpoint_id == CheckpointEvaluation.checkpoint_id)
            .order_by(
                CheckpointEvaluation.executed_at.desc(),
                CheckpointEvaluation.evaluation_id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        if checkpoint:
            stmt = stmt.where(Checkpoint.name_key == normalize_key(checkpoint))
        with self.database.read_session() as session:
            rows = session.execute(stmt).all()
        return [
            {
                "request_id": row[0],
                "status": row[1],
                "date": self._format_datetime(row[2]),
                "metrics": DatasetRepository._parse_json(row[3], default=[]),
                "results": DatasetRepository._parse_json(row[4], default={}),
                "checkpoint": row[5],
            }
            for row in rows
        ]

