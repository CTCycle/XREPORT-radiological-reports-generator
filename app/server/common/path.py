from __future__ import annotations

from pathlib import Path
import sys

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
RESOURCES_DIR = APP_DIR / "resources"
SCRIPTS_DIR = APP_DIR / "scripts"
SERVER_DIR = APP_DIR / "server"
SETTINGS_DIR = ROOT_DIR / "settings"
SHARED_DIR = APP_DIR / "shared"
TESTS_DIR = APP_DIR / "tests"
XREPORT_DIR = APP_DIR / "XREPORT"

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

CONFIGURATION_FILE_PATH = SETTINGS_DIR / "configurations.json"
DATABASE_FILE_PATH = RESOURCES_DIR / "database.db"
ENV_FILE_PATH = SETTINGS_DIR / ".env"
ENV_EXAMPLE_FILE_PATH = SETTINGS_DIR / ".env.example"
