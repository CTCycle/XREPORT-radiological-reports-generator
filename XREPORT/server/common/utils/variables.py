from __future__ import annotations

import os

from XREPORT.server.configurations.environment import load_environment


# [LOAD ENVIRONMENT VARIABLES]
###############################################################################
class EnvironmentVariables:
    def __init__(self) -> None:
        self.env_path = load_environment()

    # -------------------------------------------------------------------------
    def get(self, key: str, default: str | None = None) -> str | None:
        return os.getenv(key, default)


# Backward-compatible alias for existing imports.
def load_runtime_environment() -> None:
    load_environment()
