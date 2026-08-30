from __future__ import annotations

import os
from pathlib import Path

from server.common.path import (
    HF_HUB_CACHE_DIR,
    HF_INSTALLED_DIR,
    HF_METADATA_DIR,
    HF_ROLLBACK_DIR,
    HF_STAGING_DIR,
    HUGGINGFACE_MODELS_DIR,
    KERAS_CACHE_DIR,
    MODELS_DIR,
    TORCH_CACHE_DIR,
)


###############################################################################
def configure_model_cache() -> Path:
    """Configure all supported model libraries to use project-local storage."""
    for path in (
        MODELS_DIR,
        HUGGINGFACE_MODELS_DIR,
        HF_HUB_CACHE_DIR,
        HF_INSTALLED_DIR,
        HF_METADATA_DIR,
        HF_ROLLBACK_DIR,
        HF_STAGING_DIR,
        TORCH_CACHE_DIR,
        KERAS_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

    # Environment variables are adapters for third-party libraries. The project
    # paths above remain the source of truth and are deliberately overwritten on
    # every startup.
    os.environ["HF_HOME"] = str(HUGGINGFACE_MODELS_DIR)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
    os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
    os.environ["KERAS_HOME"] = str(KERAS_CACHE_DIR)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ.pop("HF_CACHE_DIR", None)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return HUGGINGFACE_MODELS_DIR
