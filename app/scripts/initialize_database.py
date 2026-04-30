from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from server.configurations import get_server_settings
from server.repositories.database.initializer import initialize_database
from server.common.utils.logger import logger


###############################################################################
if __name__ == "__main__":
    start = time.perf_counter()
    server_settings = get_server_settings()
    logger.info("Starting database initialization")
    logger.info(
        "Current database configuration: %s",
        json.dumps(asdict(server_settings.database), ensure_ascii=False),
    )
    initialize_database()
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds", elapsed)
