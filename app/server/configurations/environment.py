from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from ..common.path import ENV_EXAMPLE_FILE_PATH, ENV_FILE_PATH
from ..common.utils.logger import logger

###############################################################################
@dataclass
class _EnvironmentState:
    lock: Lock = field(default_factory=Lock)
    loaded: bool = False

###############################################################################
@lru_cache(maxsize=1)
def _environment_state() -> _EnvironmentState:
    return _EnvironmentState()

###############################################################################
def ensure_environment_file() -> Path:
    if ENV_FILE_PATH.exists():
        return ENV_FILE_PATH

    if not ENV_EXAMPLE_FILE_PATH.is_file():
        raise FileNotFoundError(
            f"Environment template not found: {ENV_EXAMPLE_FILE_PATH}"
        )

    ENV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with ENV_FILE_PATH.open("xb") as destination:
            destination.write(ENV_EXAMPLE_FILE_PATH.read_bytes())
        created = True
    except FileExistsError:
        # Another process created the file after the existence check. Preserve it.
        pass

    if created:
        logger.info("Created environment file from template at %s", ENV_FILE_PATH)
    return ENV_FILE_PATH

###############################################################################
def load_environment(*, force: bool = False) -> Path:
    state = _environment_state()
    with state.lock:
        env_path = ensure_environment_file()
        if state.loaded and not force:
            return env_path

        load_dotenv(dotenv_path=env_path, override=True)

        state.loaded = True
        return env_path
