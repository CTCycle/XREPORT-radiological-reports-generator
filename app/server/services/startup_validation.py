from __future__ import annotations

import os
from pathlib import Path

from server.common.path import (
    CHECKPOINT_PATH,
    CLIENT_INDEX_FILE_PATH,
    CONFIGURATION_FILE_PATH,
    LOGS_PATH,
    MODELS_PATH,
    RESOURCES_PATH,
    TEMPLATES_PATH,
    TOKENIZERS_PATH,
)
from server.common.utils.logger import logger
from server.configurations import ServerSettings, get_server_settings


def _tauri_mode_enabled() -> bool:
    value = os.getenv("XREPORT_TAURI_MODE", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_startup_validations(settings: ServerSettings | None = None) -> None:
    resolved_settings = settings or get_server_settings()

    if not CONFIGURATION_FILE_PATH.is_file():
        raise RuntimeError(f"Configuration file not found: {CONFIGURATION_FILE_PATH}")

    for directory in (
        RESOURCES_PATH,
        LOGS_PATH,
        MODELS_PATH,
        TOKENIZERS_PATH,
        CHECKPOINT_PATH,
        TEMPLATES_PATH,
    ):
        _ensure_directory(directory)

    if _tauri_mode_enabled() and not CLIENT_INDEX_FILE_PATH.is_file():
        raise RuntimeError(
            "Tauri mode requires a built frontend at "
            f"{CLIENT_INDEX_FILE_PATH}."
        )

    logger.info(
        "Startup validations completed (embedded_database=%s, tauri_mode=%s)",
        resolved_settings.database.embedded_database,
        _tauri_mode_enabled(),
    )
