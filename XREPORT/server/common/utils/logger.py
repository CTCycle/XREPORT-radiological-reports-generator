from __future__ import annotations

import logging
import logging.config
import os
from datetime import datetime
from typing import Any

from XREPORT.server.common.constants import LOGS_PATH


###############################################################################
class IgnoreJobPollingAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True

        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True

        method = str(args[1]).upper()
        path = str(args[2])

        if method == "GET" and "/jobs/" in path:
            return False

        return True


# Generate timestamp for the log filename
###############################################################################
os.makedirs(LOGS_PATH, exist_ok=True)
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(LOGS_PATH, f"XREPORT_{current_timestamp}.log")

# Define logger configuration
###############################################################################
LOG_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "ignore_job_polling_access": {
            "()": "XREPORT.server.common.utils.logger.IgnoreJobPollingAccessLogFilter",
        },
    },
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%d-%m-%Y %H:%M:%S",
        },
        "minimal": {
            "format": "[%(levelname)s] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "minimal",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "default",
            "filename": log_filename,
            "mode": "a",
        },
    },
    "loggers": {
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "filters": ["ignore_job_polling_access"],
            "propagate": False,
        },
        "matplotlib": {
            "level": "WARNING",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}


# override logger configuration and load root logger
###############################################################################
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger()
