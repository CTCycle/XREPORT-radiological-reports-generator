from __future__ import annotations

import os

from dotenv import load_dotenv

from XREPORT.app.constants import PROJECT_DIR
from XREPORT.app.logger import logger


# [IMPORT CUSTOM MODULES]
###############################################################################
class EnvironmentVariables:
    def __init__(self) -> None:
        self.env_path = os.path.join(PROJECT_DIR, "setup", ".env")
        if os.path.exists(self.env_path):
            load_dotenv(dotenv_path=self.env_path, override=True)
        else:
            logger.error(f".env file not found at: {self.env_path}")

    # -------------------------------------------------------------------------
    def get(self, key: str, default: str | None = None) -> str | None:
        return os.getenv(key, default)
