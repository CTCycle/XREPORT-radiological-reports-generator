from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
from keras import Model
from keras.models import load_model
import hashlib
import sqlalchemy
from sqlalchemy.exc import OperationalError

from XREPORT.server.utils.constants import (
    RESOURCES_PATH,
    CHECKPOINT_PATH,
    VALID_IMAGE_EXTENSIONS,
    RADIOGRAPHY_TABLE,
    TRAINING_DATASET_TABLE,
    PROCESSING_METADATA_TABLE,
    GENERATED_REPORTS_TABLE,
    TEXT_STATISTICS_TABLE,
    IMAGE_STATISTICS_TABLE,
    CHECKPOINTS_SUMMARY_TABLE,
    VALIDATION_REPORTS_TABLE,
    TABLE_REQUIRED_COLUMNS,
    TABLE_MERGE_KEYS,
)
from XREPORT.server.utils.logger import logger
from XREPORT.server.database.database import database
from XREPORT.server.database.sqlite import SQLiteRepository
from XREPORT.server.utils.learning.training.metrics import (
    MaskedSparseCategoricalCrossentropy,
    MaskedAccuracy,
)
from XREPORT.server.utils.learning.training.scheduler import WarmUpLRScheduler
from XREPORT.server.utils.learning.training.layers import (
    PositionalEmbedding,
    AddNorm,
    FeedForward,
    SoftMaxClassifier,
    TransformerEncoder,
    TransformerDecoder,
)
from XREPORT.server.utils.learning.training.encoder import BeitXRayImageEncoder

VALID_EXTENSIONS = VALID_IMAGE_EXTENSIONS


###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        self.img_shape = (224, 224)
        self.num_channels = 3
        self.valid_extensions = VALID_EXTENSIONS

    # -------------------------------------------------------------------------
    @staticmethod
    def generate_hashcode(metadata: dict) -> str:
        """Generate a deterministic hash for the dataset processing configuration."""
        if not metadata:
            return ""
        
        # Deterministic payload extraction
        payload = {
            "dataset_name": metadata.get("dataset_name"),
            "sample_size": metadata.get("sample_size"),
            "validation_size": metadata.get("validation_size"),
            "seed": metadata.get("seed"),
            "vocabulary_size": metadata.get("vocabulary_size"),
            "max_report_size": metadata.get("max_report_size"),
            "tokenizer": metadata.get("tokenizer"),
        }
        
        # Serialize to JSON with sort_keys=True
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
        """
        Identify columns containing lists or dicts and serialize them to JSON strings.
        Returns a copy of the dataframe with serialized columns.
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        for col in df_copy.columns:
            # Check first non-null value
            first_valid = df_copy[col].dropna().iloc[0] if not df_copy[col].dropna().empty else None
            
            if isinstance(first_valid, (list, dict)):
                # Serialize list/dict to JSON string
                # We use generic json.dumps. 
                # Note: This modifies the column to object/string type.
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

        dataset = database.load_from_database(table_name)
        if dataset.empty:
            return dataset

        if offset:
            dataset = dataset.iloc[offset:]
        if limit is not None:
            dataset = dataset.head(limit)

        return dataset.reset_index(drop=True)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        return database.count_rows(table_name)

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
        
        # Serialize list/dict columns to JSON strings if needed
        dataset_to_save = self._serialize_json_columns(dataset)
        database.save_into_database(dataset_to_save, table_name)

    # -------------------------------------------------------------------------
    def merge_table(
        self,
        dataset: pd.DataFrame,
        table_name: str,
        merge_keys: list[str],
    ) -> None:
        if dataset.empty:
            logger.debug("Skipping merge for %s: dataset is empty", table_name)
            return
        required_columns = TABLE_REQUIRED_COLUMNS.get(table_name)
        if required_columns:
            self.validate_required_columns(
                dataset, required_columns, table_name, "merge"
            )
        missing_merge_keys = [key for key in merge_keys if key not in dataset.columns]
        if missing_merge_keys:
            raise ValueError(
                f"Missing merge keys for {table_name}: {', '.join(missing_merge_keys)}"
            )

        existing = self.load_table(table_name)
        if existing.empty:
            merged = dataset.copy()
        else:
            merged = pd.concat([existing, dataset], ignore_index=True)
            merged = merged.drop_duplicates(subset=merge_keys, keep="last")

        # Serialize list/dict columns to JSON strings if needed
        merged_to_save = self._serialize_json_columns(merged)
        database.save_into_database(merged_to_save, table_name)

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
        try:
            # Serialize list/dict columns to JSON strings if needed
            dataset_to_save = self._serialize_json_columns(dataset)
            database.upsert_into_database(dataset_to_save, table_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Upsert failed for %s, falling back to merge/save: %s",
                table_name,
                exc,
            )
            merge_keys = TABLE_MERGE_KEYS.get(table_name, [])
            if merge_keys:
                self.merge_table(dataset, table_name, merge_keys)
            else:
                self.save_table(dataset, table_name)

    # -------------------------------------------------------------------------
    def validate_metadata(
        self, metadata: dict[str, Any] | Any, target_metadata: dict[str, Any] | Any
    ) -> bool:
        keys_to_compare = [k for k in metadata if k != "date"]
        meta_current = {k: metadata.get(k) for k in keys_to_compare}
        meta_target = {k: target_metadata.get(k) for k in keys_to_compare}
        differences = {
            k: (meta_current[k], meta_target[k])
            for k in keys_to_compare
            if meta_current[k] != meta_target[k]
        }

        return False if differences else True

    # -------------------------------------------------------------------------
    def validate_img_paths(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Validate that stored image paths exist and filter out missing ones."""
        if "path" not in dataset.columns:
            logger.error("Dataset missing 'path' column - images were not stored with paths")
            return pd.DataFrame()
        
        # Check which paths actually exist
        valid_mask = dataset["path"].apply(lambda p: os.path.isfile(p) if pd.notna(p) else False)
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
        if dataset_name and "dataset_name" in dataset.columns:
            dataset = dataset[dataset["dataset_name"] == dataset_name]
        if sample_size < 1.0:
            dataset = dataset.sample(frac=sample_size, random_state=seed)

        return dataset

    # -------------------------------------------------------------------------
    def load_training_data(
        self,
        only_metadata: bool = False,
        dataset_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict] | dict:
        # Load metadata from database
        metadata_df = self.load_table(PROCESSING_METADATA_TABLE)
        if metadata_df.empty:
            logger.warning("No processing metadata found in database")
            if only_metadata:
                return {}
            return pd.DataFrame(), pd.DataFrame(), {}

        # Filter metadata by dataset_name if provided
        if dataset_name:
            filtered_meta = metadata_df[metadata_df["dataset_name"] == dataset_name]
            if filtered_meta.empty:
                logger.warning(f"No metadata found for dataset: {dataset_name}")
                if only_metadata:
                    return {}
                return pd.DataFrame(), pd.DataFrame(), {}
            latest_metadata = filtered_meta.iloc[-1].to_dict()
        else:
            # Get the latest metadata record (last row) if no specific dataset requested
            latest_metadata = metadata_df.iloc[-1].to_dict()
            
        # Remove the 'id' column from metadata dict
        latest_metadata.pop("id", None)

        if only_metadata:
            return latest_metadata

        training_data = self.load_table(TRAINING_DATASET_TABLE)
        if training_data.empty:
            return pd.DataFrame(), pd.DataFrame(), latest_metadata
        
        # Filter training data by dataset_name if using the new schema
        if "dataset_name" in training_data.columns:
            target_name = dataset_name or latest_metadata.get("dataset_name")
            if target_name:
                training_data = training_data[training_data["dataset_name"] == target_name]

        train_data = training_data[training_data["split"] == "train"].copy()
        val_data = training_data[training_data["split"] == "validation"].copy()

        # Deserialize tokens from JSON strings if specific columns exist
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

        self.validate_required_columns(
            training_data,
            ["image", "tokens", "split"],
            TRAINING_DATASET_TABLE,
            "prepare",
        )

        db_columns = ["dataset_name", "hashcode", "id", "image", "tokens", "split", "path"]
        
        # Ensure path column exists
        if "path" not in training_data.columns:
            logger.warning("Training data missing 'path' column - adding empty paths")
            training_data["path"] = None
            
        # Add dataset_name and hashcode
        dataset_name = configuration.get("dataset_name", "default")
        training_data["dataset_name"] = dataset_name
        training_data["hashcode"] = hashcode
            
        training_data_filtered = training_data[db_columns].copy()

        required_columns = TABLE_REQUIRED_COLUMNS.get(TRAINING_DATASET_TABLE, [])
        if required_columns:
            self.validate_required_columns(
                training_data_filtered, required_columns, TRAINING_DATASET_TABLE, "save"
            )
        
        try:
            # Use upsert instead of save_table to allow multiple datasets to coexist
            self.upsert_table(training_data_filtered, TRAINING_DATASET_TABLE)
        except OperationalError as e:
            error_msg = str(e)
            if "no column named" in error_msg or "has no column" in error_msg:
                raise RuntimeError(
                    "Database schema mismatch detected. The database table structure "
                    "does not match the expected schema. Please delete the database file "
                    "(resources/database/XREPORT.db) and restart the server to reinitialize."
                ) from e
            raise
        
        # Save metadata to database table
        metadata = {
            "dataset_name": configuration.get("dataset_name", "default"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "seed": configuration.get("seed", 42),
            "sample_size": configuration.get("sample_size", 1.0),
            "validation_size": configuration.get("validation_size", 0.2),
            "vocabulary_size": vocabulary_size,
            "max_report_size": configuration.get("max_report_size", 200),
            "tokenizer": configuration.get("tokenizer", None),
            "hashcode": hashcode,
        }
        
        metadata_df = pd.DataFrame([metadata])
        try:
            self.save_table(metadata_df, PROCESSING_METADATA_TABLE)
        except OperationalError as e:
            error_msg = str(e)
            if "no column named" in error_msg or "has no column" in error_msg:
                raise RuntimeError(
                    "Database schema mismatch for PROCESSING_METADATA table. "
                    "Please delete the database file and restart the server to reinitialize."
                ) from e
            raise

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
    def save_checkpoints_summary(self, data: pd.DataFrame) -> None:
        self.upsert_table(data, CHECKPOINTS_SUMMARY_TABLE)

    # -------------------------------------------------------------------------
    def save_validation_report(self, report: dict[str, Any]) -> None:
        dataset_name = str(report.get("dataset_name") or "default")
        should_serialize_json = isinstance(database.backend, SQLiteRepository)
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
            "dataset_name": dataset_name,
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
        with database.backend.engine.connect() as conn:
            inspector = sqlalchemy.inspect(conn)
            if not inspector.has_table(VALIDATION_REPORTS_TABLE):
                return None
            reports = pd.read_sql_table(VALIDATION_REPORTS_TABLE, conn)
            if reports.empty:
                return None
        filtered = reports[reports["dataset_name"] == dataset_name]
        if filtered.empty:
            return None
        row = filtered.iloc[-1]

        metrics = DataSerializer._parse_json(
            row.get("metrics"), default=[]
        )
        text_statistics = DataSerializer._parse_json(
            row.get("text_statistics")
        )
        image_statistics = DataSerializer._parse_json(
            row.get("image_statistics")
        )
        pixel_distribution = DataSerializer._parse_json(
            row.get("pixel_distribution")
        )
        artifacts = DataSerializer._parse_json(
            row.get("artifacts")
        )
        
        return {
            "dataset_name": dataset_name,
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
        with database.backend.engine.connect() as conn:
            inspector = sqlalchemy.inspect(conn)
            if not inspector.has_table(VALIDATION_REPORTS_TABLE):
                return False
            reports = pd.read_sql_table(VALIDATION_REPORTS_TABLE, conn)
            if reports.empty or "dataset_name" not in reports.columns:
                return False
            return bool((reports["dataset_name"] == dataset_name).any())


###############################################################################
class ModelSerializer:
    def __init__(self) -> None:
        self.model_name = "XREPORT"

    # -------------------------------------------------------------------------
    def create_checkpoint_folder(self, name: str | None = None) -> str:
        if name:
            # Sanitize name: remove non-alphanumeric except underscore and hyphen
            import re
            sanitized_name = re.sub(r'[^a-zA-Z0-9_\-]', '', name)
            if not sanitized_name:
                today_datetime = datetime.now().strftime("%Y%m%dT%H%M%S")
                sanitized_name = f"{self.model_name}_{today_datetime}"
        else:
            today_datetime = datetime.now().strftime("%Y%m%dT%H%M%S")
            sanitized_name = f"{self.model_name}_{today_datetime}"
            
        checkpoint_path = os.path.join(CHECKPOINT_PATH, sanitized_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_path, "configuration"), exist_ok=True)
        logger.debug(f"Created checkpoint folder at {checkpoint_path}")

        return checkpoint_path

    # -------------------------------------------------------------------------
    def save_pretrained_model(self, model: Model, path: str) -> None:
        model_files_path = os.path.join(path, "saved_model.keras")
        model.save(model_files_path)
        logger.info(
            f"Training session is over. Model {os.path.basename(path)} has been saved"
        )

    # -------------------------------------------------------------------------
    def save_training_configuration(
        self,
        path: str,
        history: dict[str, Any],
        configuration: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        config_path = os.path.join(path, "configuration", "configuration.json")
        metadata_path = os.path.join(path, "configuration", "metadata.json")
        history_path = os.path.join(path, "configuration", "session_history.json")
        
        with open(config_path, "w") as f:
            json.dump(configuration, f)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        with open(history_path, "w") as f:
            json.dump(history, f)

        logger.debug(
            f"Model configuration, session history and metadata saved for {os.path.basename(path)}"
        )

    # -------------------------------------------------------------------------
    def load_training_configuration(self, path: str) -> tuple[dict, dict, dict]:
        config_path = os.path.join(path, "configuration", "configuration.json")
        with open(config_path) as f:
            configuration = json.load(f)

        metadata_path = os.path.join(path, "configuration", "metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        history_path = os.path.join(path, "configuration", "session_history.json")
        with open(history_path) as f:
            history = json.load(f)

        return configuration, metadata, history

    # -------------------------------------------------------------------------
    def scan_checkpoints_folder(self) -> list[str]:
        if not os.path.exists(CHECKPOINT_PATH):
            return []

        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                has_keras = any(
                    f.name.endswith(".keras") and f.is_file()
                    for f in os.scandir(entry.path)
                )
                if has_keras:
                    model_folders.append(entry.name)

        return model_folders

    # -------------------------------------------------------------------------
    def load_checkpoint(
        self, checkpoint: str, custom_objects: dict[str, Any] | None = None
    ) -> tuple[Model | Any, dict[str, Any], dict[str, Any], dict[str, Any], str]:
        """Load checkpoint model and configuration for resume training or inference."""
        
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint)
        model_path = os.path.join(checkpoint_path, "saved_model.keras")
        
        # Default custom objects for XREPORT models
        # Include all custom layers, metrics, and loss functions
        default_custom_objects = {
            # Loss and metrics
            "MaskedSparseCategoricalCrossentropy": MaskedSparseCategoricalCrossentropy,
            "MaskedAccuracy": MaskedAccuracy,
            "LRScheduler": WarmUpLRScheduler,
            # Custom layers
            "PositionalEmbedding": PositionalEmbedding,
            "AddNorm": AddNorm,
            "FeedForward": FeedForward,
            "SoftMaxClassifier": SoftMaxClassifier,
            "TransformerEncoder": TransformerEncoder,
            "TransformerDecoder": TransformerDecoder,
            "BeitXRayImageEncoder": BeitXRayImageEncoder,
        }
        if custom_objects:
            default_custom_objects.update(custom_objects)
        
        model = load_model(model_path, custom_objects=default_custom_objects)
        configuration, metadata, session = self.load_training_configuration(
            checkpoint_path
        )

        return model, configuration, metadata, session, checkpoint_path
