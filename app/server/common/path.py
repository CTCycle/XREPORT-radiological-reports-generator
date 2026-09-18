from __future__ import annotations

from pathlib import Path
from .runtime_layout import runtime_layout_from_environment


RUNTIME_LAYOUT = runtime_layout_from_environment()
ROOT_DIR = RUNTIME_LAYOUT.runtime_root
APP_DIR = ROOT_DIR / "app"
SCRIPTS_DIR = APP_DIR / "scripts"
SERVER_DIR = APP_DIR / "server"
SHARED_DIR = APP_DIR / "shared"
TESTS_DIR = APP_DIR / "tests"

ENV_FILE_PATH = RUNTIME_LAYOUT.environment_file
ENV_EXAMPLE_FILE_PATH = RUNTIME_LAYOUT.settings_template
DEFAULT_RESOURCES_DIR = APP_DIR / "resources"

PACKAGED_MODE = RUNTIME_LAYOUT.packaged
DATA_ROOT = RUNTIME_LAYOUT.data_root
RUNTIME_CACHE_DIR = RUNTIME_LAYOUT.cache_root
RELEASE_VERSION = RUNTIME_LAYOUT.release_version
RUNTIME_VARIANT = RUNTIME_LAYOUT.variant
RESOURCES_DIR = RUNTIME_LAYOUT.resources_root
LOGS_DIR = RESOURCES_DIR / "logs"
MODELS_DIR = RESOURCES_DIR / "models"
CHECKPOINTS_DIR = RESOURCES_DIR / "checkpoints"
TEMPLATES_DIR = RESOURCES_DIR / "templates"
ENCODERS_DIR = MODELS_DIR / "XRAYEncoder"
TOKENIZERS_DIR = MODELS_DIR / "tokenizers"
HUGGINGFACE_MODELS_DIR = MODELS_DIR / "huggingface"
HUGGINGFACE_CACHE_DIR = RUNTIME_CACHE_DIR / "huggingface"
HF_HUB_CACHE_DIR = HUGGINGFACE_CACHE_DIR / "hub"
HF_MODULES_CACHE_DIR = HUGGINGFACE_CACHE_DIR / "modules"
HF_DATASETS_CACHE_DIR = HUGGINGFACE_CACHE_DIR / "datasets"
HF_INSTALLED_DIR = HUGGINGFACE_MODELS_DIR / "installed"
HF_STAGING_DIR = HUGGINGFACE_MODELS_DIR / "staging"
HF_ROLLBACK_DIR = HUGGINGFACE_MODELS_DIR / "rollback"
HF_METADATA_DIR = HUGGINGFACE_MODELS_DIR / "metadata"
TORCH_CACHE_DIR = RUNTIME_CACHE_DIR / "torch"
KERAS_CACHE_DIR = RUNTIME_CACHE_DIR / "keras"
MATPLOTLIB_CACHE_DIR = RUNTIME_CACHE_DIR / "matplotlib"
UV_CACHE_DIR = RUNTIME_CACHE_DIR / "uv"
PIP_CACHE_DIR = RUNTIME_CACHE_DIR / "pip"
NPM_CACHE_DIR = RUNTIME_CACHE_DIR / "npm"
PLAYWRIGHT_BROWSERS_CACHE_DIR = RUNTIME_CACHE_DIR / "playwright-browsers"
PYTEST_CACHE_DIR = RUNTIME_CACHE_DIR / "pytest"
PYTEST_BASETEMP_DIR = RUNTIME_CACHE_DIR / "pytest-tmp"
RUFF_CACHE_DIR = RUNTIME_CACHE_DIR / "ruff"
MYPY_CACHE_DIR = RUNTIME_CACHE_DIR / "mypy"
PYTHON_CACHE_DIR = RUNTIME_CACHE_DIR / "python"
COVERAGE_CACHE_DIR = RUNTIME_CACHE_DIR / "coverage"
ANGULAR_CACHE_DIR = RUNTIME_CACHE_DIR / "angular"

DATABASE_FILE_PATH = RESOURCES_DIR / "database.db"
CLIENT_DIST_DIR = RUNTIME_LAYOUT.client_dist_dir

###############################################################################
def is_within_allowed_roots(path: Path) -> bool:
    """Return whether a user/model path is inside runtime or data storage."""
    resolved = path.resolve()
    for root in (
        RUNTIME_LAYOUT.runtime_root.resolve(),
        RUNTIME_LAYOUT.data_root.resolve(),
        RUNTIME_LAYOUT.resources_root.resolve(),
    ):
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False
