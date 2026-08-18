from __future__ import annotations

import os
from pathlib import Path
import sys

from dotenv import dotenv_values

###############################################################################
def _resolve_root() -> Path:
    """Resolve the portable application root without falling back to user caches."""
    candidates: list[Path] = [Path(__file__).resolve(), Path.cwd(), Path(sys.executable).resolve()]
    for source in candidates:
        for candidate in (source, *source.parents):
            if (
                (candidate / "settings" / "inference_models.json").is_file()
                and (candidate / "app" / "resources").is_dir()
            ):
                return candidate
    raise RuntimeError(
        "Unable to resolve the XREPORT application root; refusing to use a global model cache."
    )


ROOT_DIR = _resolve_root()
APP_DIR = ROOT_DIR / "app"
SCRIPTS_DIR = APP_DIR / "scripts"
SERVER_DIR = APP_DIR / "server"
SETTINGS_DIR = ROOT_DIR / "settings"
SHARED_DIR = APP_DIR / "shared"
TESTS_DIR = APP_DIR / "tests"

CONFIGURATION_FILE_PATH = SETTINGS_DIR / "configurations.json"
ENV_FILE_PATH = SETTINGS_DIR / ".env"
ENV_EXAMPLE_FILE_PATH = SETTINGS_DIR / ".env.example"
DEFAULT_RESOURCES_DIR = APP_DIR / "resources"

###############################################################################
def _configured_resources_dir() -> str | None:
    """Read the resource override before the normal dotenv bootstrap runs."""
    environment_path = (
        ENV_FILE_PATH if ENV_FILE_PATH.is_file() else ENV_EXAMPLE_FILE_PATH
    )
    if environment_path.is_file():
        configured = dotenv_values(environment_path).get("XREPORT_RESOURCES_DIR")
        if configured is not None:
            return configured
    return os.getenv("XREPORT_RESOURCES_DIR")

###############################################################################
def _resolve_resources_dir() -> Path:
    configured = _configured_resources_dir()
    if not configured or not configured.strip():
        return DEFAULT_RESOURCES_DIR

    resource_dir = Path(configured).expanduser()
    if not resource_dir.is_absolute():
        resource_dir = ROOT_DIR / resource_dir
    return resource_dir.resolve()


RESOURCES_DIR = _resolve_resources_dir()
LOGS_DIR = RESOURCES_DIR / "logs"
MODELS_DIR = RESOURCES_DIR / "models"
CHECKPOINTS_DIR = RESOURCES_DIR / "checkpoints"
TEMPLATES_DIR = RESOURCES_DIR / "templates"
ENCODERS_DIR = MODELS_DIR / "XRAYEncoder"
TOKENIZERS_DIR = MODELS_DIR / "tokenizers"
HUGGINGFACE_MODELS_DIR = MODELS_DIR / "huggingface"
HF_HUB_CACHE_DIR = HUGGINGFACE_MODELS_DIR / "hub-cache"
HF_INSTALLED_DIR = HUGGINGFACE_MODELS_DIR / "installed"
HF_STAGING_DIR = HUGGINGFACE_MODELS_DIR / "staging"
HF_ROLLBACK_DIR = HUGGINGFACE_MODELS_DIR / "rollback"
HF_METADATA_DIR = HUGGINGFACE_MODELS_DIR / "metadata"
TORCH_CACHE_DIR = MODELS_DIR / "torch"
KERAS_CACHE_DIR = MODELS_DIR / "keras"

DATABASE_FILE_PATH = RESOURCES_DIR / "database.db"
