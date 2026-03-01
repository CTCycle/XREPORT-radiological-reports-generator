from __future__ import annotations

from dataclasses import asdict
import json
import time

from XREPORT.server.configurations import server_settings
from XREPORT.server.repositories.database.initializer import initialize_database
from XREPORT.server.common.utils.logger import logger



###############################################################################
if __name__ == "__main__":
    start = time.perf_counter()
    logger.info("Starting database initialization")
    logger.info(
        "Current database configuration: %s",
        json.dumps(asdict(server_settings.database), ensure_ascii=False),
    )
    initialize_database()
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds", elapsed)
