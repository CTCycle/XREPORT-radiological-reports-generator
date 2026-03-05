from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from keras import Model
from keras.models import load_model

from XREPORT.server.common.constants import CHECKPOINT_PATH
from XREPORT.server.common.utils.logger import logger
from XREPORT.server.common.utils.security import validate_checkpoint_name
from XREPORT.server.learning.training.encoder import BeitXRayImageEncoder
from XREPORT.server.learning.training.layers import (
    AddNorm,
    FeedForward,
    PositionalEmbedding,
    SoftMaxClassifier,
    TransformerDecoder,
    TransformerEncoder,
)
from XREPORT.server.learning.training.metrics import (
    MaskedAccuracy,
    MaskedSparseCategoricalCrossentropy,
)
from XREPORT.server.learning.training.scheduler import WarmUpLRScheduler


###############################################################################
class ModelSerializer:
    def __init__(self) -> None:
        self.model_name = "XREPORT"

    # -------------------------------------------------------------------------
    def create_checkpoint_folder(self, name: str | None = None) -> str:
        if name:
            sanitized_name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
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
        checkpoint_name = validate_checkpoint_name(checkpoint)
        base_path = os.path.realpath(CHECKPOINT_PATH)
        checkpoint_path = os.path.realpath(os.path.join(base_path, checkpoint_name))
        if os.path.commonpath([base_path, checkpoint_path]) != base_path:
            raise ValueError("Checkpoint path is outside the checkpoints directory")
        model_path = os.path.join(checkpoint_path, "saved_model.keras")

        default_custom_objects = {
            "MaskedSparseCategoricalCrossentropy": MaskedSparseCategoricalCrossentropy,
            "MaskedAccuracy": MaskedAccuracy,
            "LRScheduler": WarmUpLRScheduler,
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
