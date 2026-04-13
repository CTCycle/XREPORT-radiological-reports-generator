from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from XREPORT.server.common.constants import ENV_FILE_PATH
from XREPORT.server.common.utils.logger import logger


@dataclass
class _EnvironmentState:
    lock: Lock = field(default_factory=Lock)
    loaded: bool = False


@lru_cache(maxsize=1)
def _environment_state() -> _EnvironmentState:
    return _EnvironmentState()


###############################################################################
def load_environment(*, force: bool = False) -> Path | None:
    state = _environment_state()
    env_path = Path(ENV_FILE_PATH)
    with state.lock:
        if state.loaded and not force:
            return env_path if env_path.exists() else None

        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            logger.warning(".env file not found at: %s", env_path)

        state.loaded = True
        return env_path if env_path.exists() else None
