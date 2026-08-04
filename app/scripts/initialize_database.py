from __future__ import annotations

import time

from server.common.utils.logger import logger
from server.configurations import get_server_settings
from server.repositories.database.initializer import initialize_database




###############################################################################
if __name__ == "__main__":
    start = time.perf_counter()
    server_settings = get_server_settings()
    logger.info(
        "Starting database initialization for %s",
        server_settings.database.backend,
    )
    initialize_database(server_settings.database)
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds", elapsed)
