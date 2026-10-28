from __future__ import annotations

import logging
import logging.config
from datetime import datetime

from xreport_shared.common.path import LOGS_DIR


###############################################################################
LOGS_DIR.mkdir(parents=True, exist_ok=True)
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = LOGS_DIR / f"XREPORT_{current_timestamp}.log"

###############################################################################
class RoutineAccessFilter(logging.Filter):
    """Demote successful HTTP access records to DEBUG so polling stays out of the console."""

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn access args: (client, method, path, http_version, status_code)
        args = record.args
        if isinstance(args, tuple) and len(args) >= 5 and isinstance(args[4], int):
            if args[4] < 400:
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
        return True


LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "routine_access": {"()": RoutineAccessFilter},
    },
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
    "loggers": {
        "uvicorn.access": {"filters": ["routine_access"]},
        "asyncio": {"level": "WARNING"},
        "PIL": {"level": "WARNING"},
        "matplotlib": {"level": "WARNING"},
        "urllib3": {"level": "WARNING"},
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "h5py": {"level": "WARNING"},
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("XREPORT")
