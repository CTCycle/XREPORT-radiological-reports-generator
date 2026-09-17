from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from threading import RLock
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from ..common.path import CONFIGURATION_FILE_PATH
from ..common.utils.logger import logger
from .settings import JsonServerSettings, ServerSettings

###############################################################################
class ConfigurationManager:

    # -------------------------------------------------------------------------
    def __init__(self, config_path: str | Path | None = None) -> None:
        self._lock = RLock()
        self._config_path = (
            Path(config_path) if config_path else CONFIGURATION_FILE_PATH
        )
        self._json_settings: JsonServerSettings | None = None
        self._server_settings: ServerSettings | None = None
        self.reload()

    # -------------------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self._config_path

    # -------------------------------------------------------------------------
    def _read_payload(self) -> dict[str, Any]:
        if not self._config_path.exists():
            raise RuntimeError(f"Configuration file not found: {self._config_path}")
        try:
            payload = json.loads(self._config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unable to load configuration from {self._config_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Configuration must be a JSON object.")
        return payload

    # -------------------------------------------------------------------------
    def reload(self, config_path: str | Path | None = None) -> ServerSettings:
        with self._lock:
            if config_path:
                self._config_path = Path(config_path)
            payload = self._read_payload()
            try:
                self._json_settings = JsonServerSettings.model_validate(payload)
            except ValidationError as exc:
                raise RuntimeError(
                    f"Invalid application configuration in {self._config_path}"
                ) from exc
            self._server_settings = self._json_settings.to_server_settings()
            return self._server_settings

    # -------------------------------------------------------------------------
    def get_all(self) -> ServerSettings:
        with self._lock:
            if self._server_settings is None:
                return self.reload()
            return self._server_settings

    # -------------------------------------------------------------------------
    def update_application_settings(
        self, patch: Mapping[str, Any]
    ) -> ServerSettings:
        """Apply a validated partial patch and persist the full JSON document."""
        with self._lock:
            if self._json_settings is None or self._server_settings is None:
                self.reload()
            if self._json_settings is None:  # pragma: no cover - defensive guard
                raise RuntimeError("Application configuration is not loaded")

            payload = self._json_settings.model_dump(mode="json", by_alias=True)
            self._merge_nested(payload, patch)
            try:
                updated_json = JsonServerSettings.model_validate(payload)
                updated_server = updated_json.to_server_settings()
            except ValidationError as exc:
                raise RuntimeError(
                    f"Invalid application configuration in {self._config_path}"
                ) from exc

            serialized = updated_json.model_dump(mode="json", by_alias=True)
            self._write_atomically(serialized)
            self._json_settings = updated_json
            self._server_settings = updated_server
            return updated_server

    # -------------------------------------------------------------------------
    @staticmethod
    def _merge_nested(
        target: dict[str, Any], patch: Mapping[str, Any]
    ) -> None:
        for key, value in patch.items():
            current = target.get(key)
            if isinstance(current, dict) and isinstance(value, Mapping):
                ConfigurationManager._merge_nested(current, value)
            else:
                target[key] = value

    # -------------------------------------------------------------------------
    def _write_atomically(self, payload: Mapping[str, Any]) -> None:
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._config_path.name}.",
                suffix=".tmp",
                dir=self._config_path.parent,
                text=True,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as file:
                json.dump(payload, file, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self._config_path)
        except (OSError, TypeError, ValueError) as exc:
            logger.exception("Failed to persist application settings")
            raise RuntimeError("Unable to persist application configuration") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Unable to remove temporary configuration file", exc_info=True)
