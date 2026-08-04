from __future__ import annotations

from .configurations.environment import load_environment
from .common.model_cache import configure_model_cache


load_environment()
configure_model_cache()
