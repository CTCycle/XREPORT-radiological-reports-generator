from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from XREPORT.server.common.constants import CONFIGURATION_FILE
from XREPORT.server.domain.settings import JsonServerSettings, ServerSettings


###############################################################################
class ConfigurationManager:
    def __init__(self, config_path: str | None = None) -> None:
        self._lock = RLock()
        self._config_path = Path(config_path or CONFIGURATION_FILE)
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
            raise RuntimeError(f"Unable to load configuration from {self._config_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Configuration must be a JSON object.")
        return payload

    # -------------------------------------------------------------------------
    def reload(self, config_path: str | None = None) -> ServerSettings:
        with self._lock:
            if config_path:
                self._config_path = Path(config_path)
            payload = self._read_payload()
            self._json_settings = JsonServerSettings.model_validate(payload)
            self._server_settings = self._json_settings.to_server_settings()
            return self._server_settings

    # -------------------------------------------------------------------------
    def get_all(self) -> ServerSettings:
        with self._lock:
            if self._server_settings is None:
                return self.reload()
            return self._server_settings

    # -------------------------------------------------------------------------
    def get_block(self, name: str) -> dict[str, Any]:
        normalized = name.strip().lower()
        if normalized == "global_settings":
            normalized = "global"

        with self._lock:
            if self._json_settings is None:
                self.reload()
            assert self._json_settings is not None
            blocks = self._json_settings.to_blocks()
            return dict(blocks.get(normalized, {}))

    # -------------------------------------------------------------------------
    def get_value(self, block: str, key: str, default: Any | None = None) -> Any:
        return self.get_block(block).get(key, default)
