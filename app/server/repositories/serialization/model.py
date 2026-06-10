from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from keras import Model
from keras.models import load_model

from server.common.path import CHECKPOINTS_DIR
from server.common.utils.logger import logger
from server.common.utils.security import validate_checkpoint_name
from server.learning.training.encoder import BeitXRayImageEncoder
from server.learning.training.layers import (
    AddNorm,
    FeedForward,
    PositionalEmbedding,
    SoftMaxClassifier,
    TransformerDecoder,
    TransformerEncoder,
)
from server.learning.training.metrics import (
    MaskedAccuracy,
    MaskedSparseCategoricalCrossentropy,
)
from server.learning.training.scheduler import WarmUpLRScheduler

###############################################################################
class ModelSerializer:

    # -------------------------------------------------------------------------
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

        checkpoint_path = CHECKPOINTS_DIR / sanitized_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (checkpoint_path / "configuration").mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created checkpoint folder at {checkpoint_path}")

        return str(checkpoint_path)

    # -------------------------------------------------------------------------
    def save_pretrained_model(self, model: Model, path: str | Path) -> None:
        checkpoint_path = Path(path)
        model_files_path = checkpoint_path / "saved_model.keras"
        model.save(model_files_path)
        logger.info(
            f"Training session is over. Model {checkpoint_path.name} has been saved"
        )

    # -------------------------------------------------------------------------
    def save_training_configuration(
        self,
        path: str | Path,
        history: dict[str, Any],
        configuration: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        checkpoint_path = Path(path)
        configuration_path = checkpoint_path / "configuration"
        config_path = configuration_path / "configuration.json"
        metadata_path = configuration_path / "metadata.json"
        history_path = configuration_path / "session_history.json"

        with config_path.open("w") as f:
            json.dump(configuration, f)
        with metadata_path.open("w") as f:
            json.dump(metadata, f)
        with history_path.open("w") as f:
            json.dump(history, f)

        logger.debug(
            f"Model configuration, session history and metadata saved for {checkpoint_path.name}"
        )

    # -------------------------------------------------------------------------
    def load_training_configuration(
        self, path: str | Path
    ) -> tuple[dict, dict, dict]:
        checkpoint_path = Path(path)
        configuration_path = checkpoint_path / "configuration"
        config_path = configuration_path / "configuration.json"
        with config_path.open() as f:
            configuration = json.load(f)

        metadata_path = configuration_path / "metadata.json"
        with metadata_path.open() as f:
            metadata = json.load(f)

        history_path = configuration_path / "session_history.json"
        with history_path.open() as f:
            history = json.load(f)

        return configuration, metadata, history

    # -------------------------------------------------------------------------
    def scan_checkpoints_folder(self) -> list[str]:
        if not CHECKPOINTS_DIR.exists():
            return []

        model_folders = []
        for entry in CHECKPOINTS_DIR.iterdir():
            if entry.is_dir():
                has_keras = any(
                    child.suffix == ".keras" and child.is_file()
                    for child in entry.iterdir()
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
        base_path = CHECKPOINTS_DIR.resolve()
        checkpoint_path = (base_path / checkpoint_name).resolve()
        if base_path not in checkpoint_path.parents and checkpoint_path != base_path:
            raise ValueError("Checkpoint path is outside the checkpoints directory")
        model_path = checkpoint_path / "saved_model.keras"

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

        return model, configuration, metadata, session, str(checkpoint_path)
