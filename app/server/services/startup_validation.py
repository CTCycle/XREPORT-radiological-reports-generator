from __future__ import annotations

from pathlib import Path
from time import perf_counter

from server.common.path import (
    ANGULAR_CACHE_DIR,
    CHECKPOINTS_DIR,
    COVERAGE_CACHE_DIR,
    HF_DATASETS_CACHE_DIR,
    LOGS_DIR,
    HF_MODULES_CACHE_DIR,
    MODELS_DIR,
    MATPLOTLIB_CACHE_DIR,
    MYPY_CACHE_DIR,
    NPM_CACHE_DIR,
    PIP_CACHE_DIR,
    PLAYWRIGHT_BROWSERS_CACHE_DIR,
    PYTHON_CACHE_DIR,
    PYTEST_BASETEMP_DIR,
    PYTEST_CACHE_DIR,
    RUNTIME_CACHE_DIR,
    RUFF_CACHE_DIR,
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
    UV_CACHE_DIR,
)
from server.common.utils.logger import logger
from server.configurations import (
    DatabaseSettings,
    ServerSettings,
    get_database_settings,
    get_server_settings,
)
from server.repositories.database.initializer import prepare_database_for_startup

###############################################################################
def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

###############################################################################
def run_startup_validations(
    settings: ServerSettings | DatabaseSettings | None = None,
) -> ServerSettings:
    started = perf_counter()
    if isinstance(settings, DatabaseSettings):
        database_settings = settings
        resolved_settings: ServerSettings | None = None
    else:
        database_settings = (
            settings.database if settings is not None else get_database_settings()
        )
        resolved_settings = settings

    prepare_database_for_startup(database_settings)
    resolved_settings = resolved_settings or get_server_settings()
    logger.info(
        "Startup phase=database_validated elapsed_ms=%.0f",
        (perf_counter() - started) * 1000,
    )

    for directory in (
        RUNTIME_CACHE_DIR,
        UV_CACHE_DIR,
        PIP_CACHE_DIR,
        NPM_CACHE_DIR,
        PLAYWRIGHT_BROWSERS_CACHE_DIR,
        PYTEST_CACHE_DIR,
        PYTEST_BASETEMP_DIR,
        RUFF_CACHE_DIR,
        MYPY_CACHE_DIR,
        PYTHON_CACHE_DIR,
        COVERAGE_CACHE_DIR,
        ANGULAR_CACHE_DIR,
        HF_DATASETS_CACHE_DIR,
        HF_MODULES_CACHE_DIR,
        MATPLOTLIB_CACHE_DIR,
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
        "Startup phase=resources_validated elapsed_ms=%.0f database_backend=%s",
        (perf_counter() - started) * 1000,
        resolved_settings.database.backend,
    )
    return resolved_settings
