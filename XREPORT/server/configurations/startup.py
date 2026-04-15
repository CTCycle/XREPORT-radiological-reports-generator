from __future__ import annotations

from functools import lru_cache

from XREPORT.server.common.constants import CONFIGURATION_FILE
from XREPORT.server.configurations.environment import load_environment
from XREPORT.server.configurations.management import ConfigurationManager
from XREPORT.server.domain.settings import ServerSettings


###############################################################################
@lru_cache(maxsize=1)
def get_configuration_manager() -> ConfigurationManager:
    load_environment()
    return ConfigurationManager(config_path=CONFIGURATION_FILE)


###############################################################################
def get_server_settings() -> ServerSettings:
    return get_configuration_manager().get_all()


###############################################################################
def reload_settings_for_tests(config_path: str | None = None) -> ServerSettings:
    load_environment(force=True)
    return get_configuration_manager().reload(config_path=config_path)
