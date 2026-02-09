from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import sqlalchemy

from XREPORT.server.common.constants import (
    CHECKPOINT_EVALUATION_REPORTS_TABLE,
    GENERATED_REPORTS_TABLE,
    IMAGE_STATISTICS_TABLE,
    PROCESSING_METADATA_TABLE,
    RADIOGRAPHY_TABLE,
    TABLE_REQUIRED_COLUMNS,
    TEXT_STATISTICS_TABLE,
    TRAINING_DATASET_TABLE,
    VALID_IMAGE_EXTENSIONS,
    VALIDATION_REPORTS_TABLE,
)
from XREPORT.server.common.utils.logger import logger
from XREPORT.server.repositories.database.sqlite import SQLiteRepository
from XREPORT.server.repositories.queries.data import DataRepositoryQueries
from XREPORT.server.repositories.queries.training import TrainingRepositoryQueries
from XREPORT.server.repositories.schemas import Base

VALID_EXTENSIONS = VALID_IMAGE_EXTENSIONS


###############################################################################
class DataSerializer:
    def __init__(
        self,
        queries: DataRepositoryQueries | None = None,
        training_queries: TrainingRepositoryQueries | None = None,
    ) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.training_queries = training_queries or TrainingRepositoryQueries(
            self.queries.database
        )
        self.img_shape = (224, 224)
        self.num_channels = 3
        self.valid_extensions = VALID_EXTENSIONS

    # -------------------------------------------------------------------------
    @staticmethod
    def generate_hashcode(metadata: dict) -> str:
        """Generate a deterministic hash for the dataset processing configuration."""
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

        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_json(value: Any, default: Any = None) -> Any:
        if default is None:
            default = {}
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return default
        if isinstance(value, (dict, list)):
            return value
        return default

    # -------------------------------------------------------------------------
    def serialize_series(self, col: list[int] | str) -> str | list[int]:
        if isinstance(col, list):
            return " ".join(map(str, col))
        if isinstance(col, str):
            return [int(f) for f in col.split() if f.strip()]
        return []

    # -------------------------------------------------------------------------
    def _serialize_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df_copy = df.copy()
        for col in df_copy.columns:
            first_valid = (
                df_copy[col].dropna().iloc[0]
                if not df_copy[col].dropna().empty
                else None
            )

            if isinstance(first_valid, (list, dict)):
                df_copy[col] = df_copy[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )
        return df_copy

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
                dataset, required_columns, table_name, "save"
            )

        dataset_to_save = self._serialize_json_columns(dataset)
        self.queries.save_table(dataset_to_save, table_name)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        if dataset.empty:
            logger.debug("Skipping upsert for %s: dataset is empty", table_name)
            return
        required_columns = TABLE_REQUIRED_COLUMNS.get(table_name)
        if required_columns:
            self.validate_required_columns(
                dataset, required_columns, table_name, "upsert"
            )
        dataset_to_save = self._serialize_json_columns(dataset)
        self.queries.upsert_table(dataset_to_save, table_name)

    # -------------------------------------------------------------------------
    def validate_metadata(
        self, metadata: dict[str, Any] | Any, target_metadata: dict[str, Any] | Any
    ) -> bool:
        meta_current = dict(metadata or {})
        meta_target = dict(target_metadata or {})

        meta_current.pop("id", None)
        meta_target.pop("id", None)
        keys_to_compare = [
            k for k in set(meta_current) | set(meta_target) if k != "date"
        ]
        differences = {
            k: (meta_current[k], meta_target[k])
            for k in keys_to_compare
            if meta_current.get(k) != meta_target.get(k)
        }

        return False if differences else True

    # -------------------------------------------------------------------------
    def validate_img_paths(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if "path" not in dataset.columns:
            logger.error(
                "Dataset missing 'path' column - images were not stored with paths"
            )
            return pd.DataFrame()

        valid_mask = dataset["path"].apply(
            lambda p: os.path.isfile(p) if pd.notna(p) else False
        )
        clean_dataset = dataset[valid_mask].reset_index(drop=True)
        dropped = len(dataset) - len(clean_dataset)

        if len(clean_dataset) > 0:
            logger.info(f"Validated image paths: {len(clean_dataset)} valid records")
        if dropped > 0:
            logger.warning(f"{dropped} records have missing or invalid image paths")

        return clean_dataset

    # -------------------------------------------------------------------------
    def get_img_path_from_directory(
        self, path: str, sample_size: float = 1.0
    ) -> list[str]:
        if not os.listdir(path):
            logger.error(f"No images found in {path}, please add them and try again.")
            return []
        else:
            logger.debug(f"Valid extensions are: {self.valid_extensions}")
            images_path = []
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
        dataset = self.load_table(RADIOGRAPHY_TABLE)
        if dataset_name:
            dataset = dataset[dataset["name"] == dataset_name]
        if sample_size < 1.0:
            dataset = dataset.sample(frac=sample_size, random_state=seed)

        return dataset

    # -------------------------------------------------------------------------
    def load_training_data(
        self,
        only_metadata: bool = False,
        dataset_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict] | dict:
        metadata_df = self.training_queries.load_training_metadata()
        if metadata_df.empty:
            logger.warning("No processing metadata found in database")
            if only_metadata:
                return {}
            return pd.DataFrame(), pd.DataFrame(), {}

        if dataset_name:
            filtered_meta = metadata_df[metadata_df["name"] == dataset_name]
            if filtered_meta.empty:
                logger.warning(f"No metadata found for dataset: {dataset_name}")
                if only_metadata:
                    return {}
                return pd.DataFrame(), pd.DataFrame(), {}
            latest_metadata = filtered_meta.iloc[-1].to_dict()
        else:
            latest_metadata = metadata_df.iloc[-1].to_dict()

        if only_metadata:
            return latest_metadata

        training_data = self.training_queries.load_training_dataset()
        if training_data.empty:
            return pd.DataFrame(), pd.DataFrame(), latest_metadata

        target_name = dataset_name or latest_metadata.get("name")
        if target_name:
            training_data = training_data[training_data["name"] == target_name]

        train_data = training_data[training_data["split"] == "train"].copy()
        val_data = training_data[training_data["split"] == "validation"].copy()

        if not train_data.empty and "tokens" in train_data.columns:
            train_data["tokens"] = train_data["tokens"].apply(
                lambda x: DataSerializer._parse_json(x, default=[])
            )
        if not val_data.empty and "tokens" in val_data.columns:
            val_data["tokens"] = val_data["tokens"].apply(
                lambda x: DataSerializer._parse_json(x, default=[])
            )

        return train_data, val_data, latest_metadata

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
            ["image", "text", "tokens", "split", "path"],
            TRAINING_DATASET_TABLE,
            "prepare",
        )

        dataset_name = str(configuration.get("dataset_name", "")).strip()
        if not dataset_name:
            raise ValueError("Training configuration must include a dataset_name.")
        source_dataset = str(configuration.get("source_dataset", "")).strip()
        if not source_dataset:
            raise ValueError("Training configuration must include a source_dataset.")

        db_columns = [
            "name",
            "hashcode",
            "image",
            "text",
            "tokens",
            "split",
            "path",
        ]
        training_data["name"] = dataset_name
        training_data["hashcode"] = hashcode
        training_data["text"] = training_data["text"].fillna("").astype(str)

        training_data_filtered = training_data[db_columns].copy()

        required_columns = TABLE_REQUIRED_COLUMNS.get(TRAINING_DATASET_TABLE, [])
        if required_columns:
            self.validate_required_columns(
                training_data_filtered, required_columns, TRAINING_DATASET_TABLE, "save"
            )

        serialized_training_data = self._serialize_json_columns(training_data_filtered)
        self.training_queries.upsert_training_dataset(serialized_training_data)

        metadata = {
            "name": dataset_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "seed": configuration.get("seed", 42),
            "sample_size": configuration.get("sample_size", 1.0),
            "validation_size": configuration.get("validation_size", 0.2),
            "vocabulary_size": vocabulary_size,
            "max_report_size": configuration.get("max_report_size", 200),
            "tokenizer": configuration.get("tokenizer", None),
            "hashcode": hashcode,
            "source_dataset": source_dataset,
        }

        metadata_df = pd.DataFrame([metadata])
        serialized_metadata = self._serialize_json_columns(metadata_df)
        self.training_queries.save_training_metadata(serialized_metadata)

    # -------------------------------------------------------------------------
    def upsert_source_dataset(self, dataset: pd.DataFrame) -> None:
        self.upsert_table(dataset, RADIOGRAPHY_TABLE)

    # -------------------------------------------------------------------------
    def save_generated_reports(self, reports: list[dict]) -> None:
        reports_dataframe = pd.DataFrame(reports)
        self.upsert_table(reports_dataframe, GENERATED_REPORTS_TABLE)

    # -------------------------------------------------------------------------
    def save_text_statistics(self, data: pd.DataFrame) -> None:
        self.upsert_table(data, TEXT_STATISTICS_TABLE)

    # -------------------------------------------------------------------------
    def save_images_statistics(self, data: pd.DataFrame) -> None:
        self.upsert_table(data, IMAGE_STATISTICS_TABLE)

    # -------------------------------------------------------------------------
    def save_validation_report(self, report: dict[str, Any]) -> None:
        dataset_name = str(report.get("dataset_name") or "default")
        should_serialize_json = isinstance(self.queries.backend, SQLiteRepository)
        metrics = report.get("metrics") or []
        text_statistics = report.get("text_statistics")
        image_statistics = report.get("image_statistics")
        pixel_distribution = report.get("pixel_distribution")
        artifacts = report.get("artifacts")
        if should_serialize_json:
            if isinstance(metrics, (list, dict)):
                metrics = json.dumps(metrics)
            if isinstance(text_statistics, (list, dict)):
                text_statistics = json.dumps(text_statistics)
            if isinstance(image_statistics, (list, dict)):
                image_statistics = json.dumps(image_statistics)
            if isinstance(pixel_distribution, (list, dict)):
                pixel_distribution = json.dumps(pixel_distribution)
            if isinstance(artifacts, (list, dict)):
                artifacts = json.dumps(artifacts)
        record = {
            "name": dataset_name,
            "date": report.get("date"),
            "sample_size": report.get("sample_size"),
            "metrics": metrics,
            "text_statistics": text_statistics,
            "image_statistics": image_statistics,
            "pixel_distribution": pixel_distribution,
            "artifacts": artifacts,
        }
        report_df = pd.DataFrame([record])
        self.upsert_table(report_df, VALIDATION_REPORTS_TABLE)

    # -------------------------------------------------------------------------
    def get_validation_report(self, dataset_name: str) -> dict[str, Any] | None:
        with self.queries.backend.engine.connect() as conn:
            inspector = sqlalchemy.inspect(conn)
            if not inspector.has_table(VALIDATION_REPORTS_TABLE):
                return None
            reports = pd.read_sql_table(VALIDATION_REPORTS_TABLE, conn)
            if reports.empty:
                return None
        filtered = reports[reports["name"] == dataset_name]
        if filtered.empty:
            return None
        if "id" in filtered.columns:
            filtered = filtered.sort_values(by="id")
        row = filtered.iloc[-1]

        metrics = DataSerializer._parse_json(row.get("metrics"), default=[])
        text_statistics = DataSerializer._parse_json(row.get("text_statistics"))
        image_statistics = DataSerializer._parse_json(row.get("image_statistics"))
        pixel_distribution = DataSerializer._parse_json(row.get("pixel_distribution"))
        artifacts = DataSerializer._parse_json(row.get("artifacts"))

        return {
            "dataset_name": row["name"],
            "date": row.get("date") if "date" in row else None,
            "sample_size": row.get("sample_size"),
            "metrics": metrics if isinstance(metrics, list) else [],
            "text_statistics": text_statistics,
            "image_statistics": image_statistics,
            "pixel_distribution": pixel_distribution,
            "artifacts": artifacts,
        }

    # -------------------------------------------------------------------------
    def validation_report_exists(self, dataset_name: str) -> bool:
        with self.queries.backend.engine.connect() as conn:
            inspector = sqlalchemy.inspect(conn)
            if not inspector.has_table(VALIDATION_REPORTS_TABLE):
                return False
            reports = pd.read_sql_table(VALIDATION_REPORTS_TABLE, conn)
            if reports.empty:
                return False
            return bool((reports["name"] == dataset_name).any())

    # -------------------------------------------------------------------------
    def save_checkpoint_evaluation_report(self, report: dict[str, Any]) -> None:
        checkpoint = str(report.get("checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError("Checkpoint evaluation report requires a checkpoint name")

        with self.queries.backend.engine.connect() as conn:
            inspector = sqlalchemy.inspect(conn)
            if not inspector.has_table(CHECKPOINT_EVALUATION_REPORTS_TABLE):
                Base.metadata.create_all(self.queries.backend.engine)

        should_serialize_json = isinstance(self.queries.backend, SQLiteRepository)
        metrics = report.get("metrics") or []
        metric_configs = report.get("metric_configs")
        results = report.get("results")

        if should_serialize_json:
            if isinstance(metrics, (list, dict)):
                metrics = json.dumps(metrics)
            if isinstance(metric_configs, (list, dict)):
                metric_configs = json.dumps(metric_configs)
            if isinstance(results, (list, dict)):
                results = json.dumps(results)

        record = {
            "checkpoint": checkpoint,
            "date": report.get("date"),
            "metrics": metrics,
            "metric_configs": metric_configs,
            "results": results,
        }
        report_df = pd.DataFrame([record])
        self.upsert_table(report_df, CHECKPOINT_EVALUATION_REPORTS_TABLE)

    # -------------------------------------------------------------------------
    def get_checkpoint_evaluation_report(
        self, checkpoint: str
    ) -> dict[str, Any] | None:
        with self.queries.backend.engine.connect() as conn:
            inspector = sqlalchemy.inspect(conn)
            if not inspector.has_table(CHECKPOINT_EVALUATION_REPORTS_TABLE):
                return None
            reports = pd.read_sql_table(CHECKPOINT_EVALUATION_REPORTS_TABLE, conn)
            if reports.empty:
                return None
        filtered = reports[reports["checkpoint"] == checkpoint]
        if filtered.empty:
            return None
        if "id" in filtered.columns:
            filtered = filtered.sort_values(by="id")
        row = filtered.iloc[-1]

        metrics = DataSerializer._parse_json(row.get("metrics"), default=[])
        metric_configs = DataSerializer._parse_json(
            row.get("metric_configs"), default={}
        )
        results = DataSerializer._parse_json(row.get("results"), default={})

        return {
            "checkpoint": checkpoint,
            "date": row.get("date") if "date" in row else None,
            "metrics": metrics if isinstance(metrics, list) else [],
            "metric_configs": metric_configs
            if isinstance(metric_configs, dict)
            else {},
            "results": results if isinstance(results, dict) else {},
        }

    # -------------------------------------------------------------------------
    def checkpoint_evaluation_report_exists(self, checkpoint: str) -> bool:
        with self.queries.backend.engine.connect() as conn:
            inspector = sqlalchemy.inspect(conn)
            if not inspector.has_table(CHECKPOINT_EVALUATION_REPORTS_TABLE):
                return False
            reports = pd.read_sql_table(CHECKPOINT_EVALUATION_REPORTS_TABLE, conn)
            if reports.empty:
                return False
            return bool((reports["checkpoint"] == checkpoint).any())
