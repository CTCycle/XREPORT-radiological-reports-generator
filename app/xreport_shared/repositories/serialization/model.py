from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from xreport_shared.common.path import CHECKPOINTS_DIR
from xreport_shared.common.utils.logger import logger

###############################################################################
class ModelSerializer:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.model_name = "XREPORT"

    # -------------------------------------------------------------------------
    def checkpoint_folder_name(self, name: str | None = None) -> str:
        sanitized_name = re.sub(r"[^a-zA-Z0-9_\-]", "", name) if name else ""
        if not sanitized_name:
            today_datetime = datetime.now().strftime("%Y%m%dT%H%M%S")
            sanitized_name = f"{self.model_name}_{today_datetime}"
        return sanitized_name

    # -------------------------------------------------------------------------
    def create_checkpoint_folder(self, name: str | None = None) -> str:
        checkpoint_path = CHECKPOINTS_DIR / self.checkpoint_folder_name(name)
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        # A new run never writes into an existing checkpoint.
        try:
            checkpoint_path.mkdir()
        except FileExistsError as exc:
            raise ValueError(
                f"A checkpoint named '{checkpoint_path.name}' already exists"
            ) from exc
        (checkpoint_path / "configuration").mkdir()
        logger.debug(f"Created checkpoint folder at {checkpoint_path}")

        return str(checkpoint_path)

    # -------------------------------------------------------------------------
    def save_pretrained_model(self, model: Any, path: str | Path) -> None:
        checkpoint_path = Path(path)
        model_files_path = checkpoint_path / "saved_model.keras"
        # Keras requires the .keras suffix, so the temporary name keeps it.
        temporary_path = checkpoint_path / ".saved_model.tmp.keras"
        # The zipped writer builds the weights file in memory, which fails for
        # models above 2 GB. Write the unzipped layout to disk and store it as
        # a regular .keras archive instead.
        with tempfile.TemporaryDirectory(dir=checkpoint_path) as staging:
            staged = Path(staging) / "model"
            model.save(staged, zipped=False)
            with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_STORED) as archive:
                for file in sorted(staged.rglob("*")):
                    if file.is_file():
                        archive.write(file, file.relative_to(staged).as_posix())
        os.replace(temporary_path, model_files_path)
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

        for target, payload in (
            (config_path, configuration),
            (metadata_path, metadata),
            (history_path, history),
        ):
            temporary_path = target.with_name(f".{target.name}.tmp")
            with temporary_path.open("w") as f:
                json.dump(payload, f)
            os.replace(temporary_path, target)

        logger.debug(
            f"Model configuration, session history and metadata saved for {checkpoint_path.name}"
        )

    # -------------------------------------------------------------------------
    def load_training_configuration(
        self, path: str | Path
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    def load_checkpoint(
        self, checkpoint_path: str | Path, custom_objects: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], str]:
        """Load a registered checkpoint artifact from its resolved filesystem path."""
        from keras.models import load_model

        from xreport_shared.models.training.encoder import BeitXRayImageEncoder
        from xreport_shared.models.training.layers import (
            AddNorm,
            FeedForward,
            PositionalEmbedding,
            SoftMaxClassifier,
            TransformerDecoder,
            TransformerEncoder,
        )
        from xreport_shared.models.training.metrics import (
            MaskedAccuracy,
            MaskedSparseCategoricalCrossentropy,
        )
        from xreport_shared.models.training.scheduler import WarmUpLRScheduler

        base_path = CHECKPOINTS_DIR.resolve()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
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
