from __future__ import annotations

from functools import lru_cache

from .environment import load_environment
from .settings import (
    DatabaseSettings,
    ServerSettings,
    application_settings_to_server_settings,
    database_settings_from_environment,
)


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    """Load only environment-owned database settings."""

    load_environment()
    return database_settings_from_environment()


def get_server_settings() -> ServerSettings:
    """Compose environment database settings with persisted application values."""

    from server.repositories.application_settings import ApplicationSettingsRepository

    values = ApplicationSettingsRepository().get_settings()
    return application_settings_to_server_settings(values, get_database_settings())


def reload_settings_for_tests() -> ServerSettings:
    load_environment(force=True)
    get_database_settings.cache_clear()
    from server.repositories.database.backend import get_database
    from server.services.settings import get_settings_service

    get_database.cache_clear()
    get_settings_service.cache_clear()
    return get_server_settings()


__all__ = [
    "get_database_settings",
    "get_server_settings",
    "reload_settings_for_tests",
]
