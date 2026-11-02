from __future__ import annotations

from functools import lru_cache

from .environment import load_environment
from .database_config import load_database_settings
from .settings import DatabaseSettings


###############################################################################
@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    """Load the independent, file-backed database connection configuration."""

    load_environment()
    return load_database_settings()


__all__ = ["get_database_settings"]
