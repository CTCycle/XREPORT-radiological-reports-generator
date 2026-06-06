from __future__ import annotations

from functools import lru_cache

from ..common.path import CONFIGURATION_FILE_PATH
from ..domain.settings import ServerSettings
from .environment import load_environment
from .management import ConfigurationManager


###############################################################################
@lru_cache(maxsize=1)
def get_configuration_manager() -> ConfigurationManager:
    load_environment()
    return ConfigurationManager(config_path=CONFIGURATION_FILE_PATH)


###############################################################################
def get_server_settings() -> ServerSettings:
    return get_configuration_manager().get_all()


###############################################################################
def reload_settings_for_tests(config_path: str | None = None) -> ServerSettings:
    load_environment(force=True)
    return get_configuration_manager().reload(config_path=config_path)
