from __future__ import annotations

from pathlib import Path

from server.common.path import (
    CHECKPOINTS_DIR,
    CONFIGURATION_FILE_PATH,
    LOGS_DIR,
    MODELS_DIR,
    RESOURCES_DIR,
    TEMPLATES_DIR,
    TOKENIZERS_DIR,
    HF_HUB_CACHE_DIR,
    HF_INSTALLED_DIR,
    HF_METADATA_DIR,
    HF_ROLLBACK_DIR,
    HF_STAGING_DIR,
    HUGGINGFACE_MODELS_DIR,
    KERAS_CACHE_DIR,
    TORCH_CACHE_DIR,
)
from server.common.utils.logger import logger
from server.configurations import ServerSettings, get_server_settings
from server.repositories.database.initializer import prepare_database_for_startup

###############################################################################
def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

###############################################################################
def run_startup_validations(settings: ServerSettings | None = None) -> None:
    resolved_settings = settings or get_server_settings()

    prepare_database_for_startup(resolved_settings.database)

    if not CONFIGURATION_FILE_PATH.is_file():
        raise RuntimeError(f"Configuration file not found: {CONFIGURATION_FILE_PATH}")

    for directory in (
        RESOURCES_DIR,
        LOGS_DIR,
        MODELS_DIR,
        TOKENIZERS_DIR,
        CHECKPOINTS_DIR,
        TEMPLATES_DIR,
        HUGGINGFACE_MODELS_DIR,
        HF_HUB_CACHE_DIR,
        HF_INSTALLED_DIR,
        HF_METADATA_DIR,
        HF_ROLLBACK_DIR,
        HF_STAGING_DIR,
        TORCH_CACHE_DIR,
        KERAS_CACHE_DIR,
    ):
        _ensure_directory(directory)

    logger.info(
        "Startup validations completed (database_backend=%s)",
        resolved_settings.database.backend,
    )
