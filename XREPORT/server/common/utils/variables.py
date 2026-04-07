from __future__ import annotations

import os

from dotenv import load_dotenv

from XREPORT.server.common.constants import ENV_FILE_PATH
from XREPORT.server.common.utils.logger import logger


###############################################################################
def load_runtime_environment() -> None:
    if os.path.exists(ENV_FILE_PATH):
        # Keep process env authoritative (already-exported vars win over file values).
        load_dotenv(dotenv_path=ENV_FILE_PATH, override=False)
    else:
        logger.warning(".env file not found at: %s", ENV_FILE_PATH)


load_runtime_environment()
