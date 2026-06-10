from __future__ import annotations

import logging
import logging.config
from datetime import datetime

from server.common.path import LOGS_DIR


###############################################################################
LOGS_DIR.mkdir(parents=True, exist_ok=True)
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = LOGS_DIR / f"XREPORT_{current_timestamp}.log"

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%d-%m-%Y %H:%M:%S",
        },
        "minimal": {
            "format": "%(levelname)s - %(message)s",
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
            "filename": str(log_filename),
            "mode": "a",
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("XREPORT")

