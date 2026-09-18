from __future__ import annotations

import os
import sys
from pathlib import Path

from server.common.path import (
    ANGULAR_CACHE_DIR,
    COVERAGE_CACHE_DIR,
    HF_DATASETS_CACHE_DIR,
    HF_HUB_CACHE_DIR,
    HF_INSTALLED_DIR,
    HF_MODULES_CACHE_DIR,
    HF_METADATA_DIR,
    HF_ROLLBACK_DIR,
    HF_STAGING_DIR,
    HUGGINGFACE_CACHE_DIR,
    HUGGINGFACE_MODELS_DIR,
    KERAS_CACHE_DIR,
    MATPLOTLIB_CACHE_DIR,
    MODELS_DIR,
    MYPY_CACHE_DIR,
    NPM_CACHE_DIR,
    PIP_CACHE_DIR,
    PLAYWRIGHT_BROWSERS_CACHE_DIR,
    PYTHON_CACHE_DIR,
    PYTEST_BASETEMP_DIR,
    PYTEST_CACHE_DIR,
    RUNTIME_CACHE_DIR,
    RUFF_CACHE_DIR,
    TORCH_CACHE_DIR,
    UV_CACHE_DIR,
)

###############################################################################
def configure_model_cache() -> Path:
    """Configure every disposable cache through the canonical runtime root."""
    for path in (
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
        HUGGINGFACE_CACHE_DIR,
        HF_HUB_CACHE_DIR,
        HF_MODULES_CACHE_DIR,
        HF_DATASETS_CACHE_DIR,
        TORCH_CACHE_DIR,
        KERAS_CACHE_DIR,
        MATPLOTLIB_CACHE_DIR,
        MODELS_DIR,
        HUGGINGFACE_MODELS_DIR,
        HF_INSTALLED_DIR,
        HF_METADATA_DIR,
        HF_ROLLBACK_DIR,
        HF_STAGING_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

    # Environment variables are adapters for third-party libraries. The project
    # paths above remain the source of truth and are deliberately overwritten on
    # every startup.
    os.environ["XREPORT_CACHE_ROOT"] = str(RUNTIME_CACHE_DIR)
    os.environ["XDG_CACHE_HOME"] = str(RUNTIME_CACHE_DIR)
    os.environ["UV_CACHE_DIR"] = str(UV_CACHE_DIR)
    os.environ["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)
    os.environ["NPM_CONFIG_CACHE"] = str(NPM_CACHE_DIR)
    os.environ["npm_config_cache"] = str(NPM_CACHE_DIR)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(PLAYWRIGHT_BROWSERS_CACHE_DIR)
    os.environ["PYTEST_CACHE_DIR"] = str(PYTEST_CACHE_DIR)
    os.environ["PYTEST_BASETEMP"] = str(PYTEST_BASETEMP_DIR)
    os.environ["RUFF_CACHE_DIR"] = str(RUFF_CACHE_DIR)
    os.environ["MYPY_CACHE_DIR"] = str(MYPY_CACHE_DIR)
    os.environ["COVERAGE_FILE"] = str(COVERAGE_CACHE_DIR / ".coverage")
    os.environ["HF_HOME"] = str(HUGGINGFACE_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
    os.environ["HF_MODULES_CACHE"] = str(HF_MODULES_CACHE_DIR)
    os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE_DIR)
    os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
    os.environ["KERAS_HOME"] = str(KERAS_CACHE_DIR)
    os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE_DIR)
    os.environ["PYTHONPYCACHEPREFIX"] = str(PYTHON_CACHE_DIR)
    sys.pycache_prefix = str(PYTHON_CACHE_DIR)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ.pop("HF_CACHE_DIR", None)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return HUGGINGFACE_CACHE_DIR
