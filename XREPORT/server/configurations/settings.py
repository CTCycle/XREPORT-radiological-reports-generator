from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

from pydantic import ValidationError

from XREPORT.server.configurations.bootstrap import ensure_environment_loaded
from XREPORT.server.domain.settings import AppSettings, ServerSettings


def _build_path_scoped_settings_class(config_path: str) -> type[AppSettings]:
    class PathScopedAppSettings(AppSettings):
        _configuration_file: ClassVar[str] = config_path

    return PathScopedAppSettings


# -----------------------------------------------------------------------------
def _load_app_settings(settings_cls: type[AppSettings]) -> AppSettings:
    ensure_environment_loaded()
    try:
        return settings_cls()
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application settings: {exc}") from exc


###############################################################################
@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return _load_app_settings(AppSettings)


# -----------------------------------------------------------------------------
def get_server_settings(config_path: str | None = None) -> ServerSettings:
    if config_path:
        scoped_class = _build_path_scoped_settings_class(config_path=config_path)
        return _load_app_settings(scoped_class).to_server_settings()
    return get_app_settings().to_server_settings()


# -----------------------------------------------------------------------------
def reload_settings_for_tests() -> AppSettings:
    get_app_settings.cache_clear()
    return get_app_settings()
