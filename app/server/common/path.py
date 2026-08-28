from __future__ import annotations

import os
from pathlib import Path
import sys

from dotenv import dotenv_values

###############################################################################
def _desktop_enabled() -> bool:
    return os.getenv("XREPORT_DESKTOP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

###############################################################################
def _required_desktop_path(name: str) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set for packaged XREPORT runtime")
    return Path(value).expanduser().resolve()

###############################################################################
def _resolve_root() -> Path:
    """Resolve the portable application root without falling back to user caches."""
    if _desktop_enabled():
        root = _required_desktop_path("XREPORT_RUNTIME_ROOT")
        if not root.is_dir():
            raise RuntimeError(f"Packaged XREPORT runtime root does not exist: {root}")
        return root
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

# Packaged mode deliberately separates immutable extracted files from mutable
# per-user state.  Source mode keeps the historical resource-root override.
PACKAGED_MODE = _desktop_enabled()
DATA_ROOT = (
    _required_desktop_path("XREPORT_DATA_ROOT") if PACKAGED_MODE else DEFAULT_RESOURCES_DIR
)
RELEASE_VERSION = os.getenv("XREPORT_RELEASE_VERSION", "").strip() or None
RUNTIME_VARIANT = os.getenv("XREPORT_RUNTIME_VARIANT", "").strip().lower() or None

###############################################################################
def _configured_resources_dir() -> str | None:
    """Read the resource override before the normal dotenv bootstrap runs."""
    if PACKAGED_MODE:
        # A packaged build must never escape its user-data boundary through a
        # source-relative XREPORT_RESOURCES_DIR setting.
        return None
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


RESOURCES_DIR = DATA_ROOT if PACKAGED_MODE else _resolve_resources_dir()
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
CLIENT_DIST_DIR = (
    Path(os.getenv("XREPORT_CLIENT_DIST_DIR", "")).expanduser().resolve()
    if PACKAGED_MODE and os.getenv("XREPORT_CLIENT_DIST_DIR", "").strip()
    else ROOT_DIR / "client"
    if PACKAGED_MODE
    else ROOT_DIR / "app" / "client" / "dist" / "client-angular" / "browser"
)

# Configuration is mutable in packaged mode but the catalogue/template files
# remain part of the verified runtime archive.
if PACKAGED_MODE:
    CONFIGURATION_FILE_PATH = DATA_ROOT / "settings" / "configurations.json"
    ENV_FILE_PATH = DATA_ROOT / ".env"

###############################################################################
def is_within_allowed_roots(path: Path) -> bool:
    """Return whether a user/model path is inside runtime or data storage."""
    resolved = path.resolve()
    for root in (ROOT_DIR.resolve(), DATA_ROOT.resolve()):
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False
