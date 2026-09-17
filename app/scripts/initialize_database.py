from __future__ import annotations

import time

from server.common.utils.logger import logger
from server.configurations import get_database_settings
from server.repositories.database.initializer import initialize_database


###############################################################################
if __name__ == "__main__":
    start = time.perf_counter()
    database_settings = get_database_settings()
    logger.info(
        "Starting database initialization for %s",
        database_settings.backend,
    )
    initialize_database(database_settings)
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds", elapsed)
