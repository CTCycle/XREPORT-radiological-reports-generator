from __future__ import annotations

from functools import lru_cache

from .environment import load_environment
from .database_config import load_database_settings
from .settings import (
    DatabaseSettings,
    InferenceSettings,
    ServerSettings,
    application_settings_to_server_settings,
)

###############################################################################
@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    """Load the independent, file-backed database connection configuration."""

    load_environment()
    return load_database_settings()

###############################################################################
def get_server_settings() -> ServerSettings:
    """Compose environment database settings with persisted application values."""

    from xreport_shared.repositories.application_settings import ApplicationSettingsRepository

    values = ApplicationSettingsRepository().get_settings()
    return application_settings_to_server_settings(values, get_database_settings())

###############################################################################
def get_inference_settings() -> InferenceSettings:
    """Return the persisted inference controls of the XREPORT Inference database."""

    inference = get_server_settings().inference
    if inference is None:
        raise RuntimeError("Inference settings exist only in the XREPORT Inference database")
    return inference

###############################################################################
def reload_settings_for_tests() -> ServerSettings:
    load_environment(force=True)
    get_database_settings.cache_clear()
    from xreport_shared.repositories.database.backend import reset_database
    from xreport_shared.services.settings import get_settings_service

    reset_database()
    get_settings_service.cache_clear()
    return get_server_settings()


__all__ = [
    "get_database_settings",
    "get_inference_settings",
    "get_server_settings",
    "reload_settings_for_tests",
]
