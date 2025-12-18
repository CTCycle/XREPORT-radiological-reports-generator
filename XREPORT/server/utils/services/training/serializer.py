from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from keras import Model
from keras.models import load_model

from XREPORT.server.utils.constants import RESOURCES_PATH
from XREPORT.server.utils.logger import logger
from XREPORT.server.database.database import database

CHECKPOINT_PATH = os.path.join(RESOURCES_PATH, "checkpoints")
IMG_PATH = os.path.join(RESOURCES_PATH, "images")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        self.img_shape = (224, 224)
        self.num_channels = 3
        self.valid_extensions = VALID_EXTENSIONS

    # -------------------------------------------------------------------------
    def serialize_series(self, col: list[str] | str) -> str | list[int]:
        if isinstance(col, list):
            return " ".join(map(str, col))
        if isinstance(col, str):
            return [int(f) for f in col.split() if f.strip()]
        return []

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
    def update_img_path(self, dataset: pd.DataFrame) -> pd.DataFrame:
        images_path = {}
        for root, _, files in os.walk(IMG_PATH):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    path_pair = {file.split(".")[0]: os.path.join(IMG_PATH, file)}
                    images_path.update(path_pair)

        dataset["path"] = dataset["image"].map(images_path)
        clean_dataset = dataset.dropna(subset=["path"]).reset_index(drop=True)
        logger.info(
            f"Updated dataset with images paths: {len(clean_dataset)} records found"
        )
        logger.info(
            f"{len(dataset) - len(clean_dataset)} records were dropped due to missing images"
        )

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
        self, sample_size: float = 1.0, seed: int = 42
    ) -> pd.DataFrame:
        dataset = database.load_from_database("RADIOGRAPHY_DATA")
        if sample_size < 1.0:
            dataset = dataset.sample(frac=sample_size, random_state=seed)

        return dataset

    # -------------------------------------------------------------------------
    def load_training_data(
        self, only_metadata: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict] | dict:
        # Load metadata from database
        metadata_df = database.load_from_database("PROCESSING_METADATA")
        if metadata_df.empty:
            logger.warning("No processing metadata found in database")
            if only_metadata:
                return {}
            return pd.DataFrame(), pd.DataFrame(), {}

        # Get the latest metadata record (last row)
        latest_metadata = metadata_df.iloc[-1].to_dict()
        # Remove the 'id' column from metadata dict
        latest_metadata.pop("id", None)

        if only_metadata:
            return latest_metadata

        training_data = database.load_from_database("TRAINING_DATASET")
        if training_data.empty:
            return pd.DataFrame(), pd.DataFrame(), latest_metadata

        training_data["tokens"] = training_data["tokens"].apply(
            self.serialize_series
        )
        train_data = training_data[training_data["split"] == "train"]
        val_data = training_data[training_data["split"] == "validation"]

        return train_data, val_data, latest_metadata

    # -------------------------------------------------------------------------
    def save_training_data(
        self,
        configuration: dict[str, Any],
        training_data: pd.DataFrame,
        vocabulary_size: int | None = None,
    ) -> None:
        training_data["tokens"] = training_data["tokens"].apply(self.serialize_series)
        database.save_into_database(training_data, "TRAINING_DATASET")
        
        # Save metadata to database table
        metadata = {
            "dataset_name": configuration.get("dataset_name", "default"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "seed": configuration.get("seed", 42),
            "sample_size": configuration.get("sample_size", 1.0),
            "validation_size": configuration.get("validation_size", 0.2),
            "split_seed": configuration.get("split_seed", 42),
            "vocabulary_size": vocabulary_size,
            "max_report_size": configuration.get("max_report_size", 200),
            "tokenizer": configuration.get("tokenizer", None),
        }
        
        metadata_df = pd.DataFrame([metadata])
        database.upsert_into_database(metadata_df, "PROCESSING_METADATA")

    # -------------------------------------------------------------------------
    def save_generated_reports(self, reports: list[dict]) -> None:
        reports_dataframe = pd.DataFrame(reports)
        database.upsert_into_database(reports_dataframe, "GENERATED_REPORTS")

    # -------------------------------------------------------------------------
    def save_text_statistics(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "TEXT_STATISTICS")

    # -------------------------------------------------------------------------
    def save_images_statistics(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "IMAGE_STATISTICS")

    # -------------------------------------------------------------------------
    def save_checkpoints_summary(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "CHECKPOINTS_SUMMARY")


###############################################################################
class ModelSerializer:
    def __init__(self) -> None:
        self.model_name = "XREPORT"

    # -------------------------------------------------------------------------
    def create_checkpoint_folder(self) -> str:
        today_datetime = datetime.now().strftime("%Y%m%dT%H%M%S")
        checkpoint_path = os.path.join(
            CHECKPOINT_PATH, f"{self.model_name}_{today_datetime}"
        )
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
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint)
        model_path = os.path.join(checkpoint_path, "saved_model.keras")
        
        if custom_objects is None:
            custom_objects = {}
        
        model = load_model(model_path, custom_objects=custom_objects)
        configuration, metadata, session = self.load_training_configuration(
            checkpoint_path
        )

        return model, configuration, metadata, session, checkpoint_path
