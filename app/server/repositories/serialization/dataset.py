from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, overload

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from server.common.constants import (
    DATASET_RECORDS_TABLE,
    TRAINING_SAMPLES_TABLE,
    VALID_IMAGE_EXTENSIONS,
)
from server.common.utils.logger import logger
from server.repositories.database import Database
from server.repositories.schemas import (
    Dataset,
    DatasetRecord,
    DatasetVersion,
    ProcessingRun,
    TrainingSample,
)
from server.repositories.schemas.normalization import normalize_key
from server.repositories.serialization.support import RepositorySupport

VALID_EXTENSIONS = VALID_IMAGE_EXTENSIONS

###############################################################################
class DatasetRepository(RepositorySupport):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        database: Database | None = None,
    ) -> None:
        super().__init__(database)
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
        clean_dataset = dataset.loc[valid_mask, :].copy().reset_index(drop=True)
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
    @overload
    def load_training_data(
        self,
        only_metadata: Literal[True],
        dataset_name: str | None = None,
    ) -> dict[str, Any]: ...

    # -------------------------------------------------------------------------
    @overload
    def load_training_data(
        self,
        only_metadata: Literal[False] = False,
        dataset_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]: ...

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
        train_data = training_data.loc[
            training_data["split"] == "train", :
        ].copy()
        val_data = training_data.loc[
            training_data["split"] == "validation", :
        ].copy()
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
        record_ids = pd.Series(
            pd.to_numeric(
            training_payload["record_id"],
            errors="coerce",
            ),
            index=training_payload.index,
        )
        missing_record_ids = int(record_ids.isna().sum())
        if missing_record_ids:
            raise ValueError(
                f"Training payload has {missing_record_ids} rows without record_id"
            )
        training_payload["record_id"] = record_ids.astype(int)

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
                (~record_ids.astype(int).isin(list(source_record_ids))).sum()
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
            training_records = training_payload.assign(
                split=normalized_split
            ).to_dict("records")
            session.add_all(
                TrainingSample(
                    processing_run_id=run.processing_run_id,
                    record_id=int(row["record_id"]),
                    split=str(row["split"]),
                    tokens_json=row["tokens"],
                )
                for row in training_records
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

                records = pd.DataFrame(group).copy().reset_index(drop=True)
                if "row_order" not in records.columns:
                    records["row_order"] = range(1, len(records) + 1)
                canonical_records = [
                    {
                        "image_name": str(row["image_name"]),
                        "image_name_key": normalize_key(str(row["image_name"])),
                        "report_text": str(row["report_text"]),
                        "image_path": str(row["image_path"]),
                        "row_order": int(row["row_order"]),
                    }
                    for row in records.to_dict("records")
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

