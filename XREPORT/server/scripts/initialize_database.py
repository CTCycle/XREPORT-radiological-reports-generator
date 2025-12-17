from __future__ import annotations

import json
import time

from APP.server.database.initializer import initialize_database
from APP.server.utils.constants import SERVER_CONFIGURATION_FILE
from APP.server.utils.logger import logger


# -----------------------------------------------------------------------------
def load_database_config() -> dict[str, object]:
    try:
        with open(SERVER_CONFIGURATION_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        logger.warning("Server configuration not found at %s", SERVER_CONFIGURATION_FILE)
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Unable to read database configuration at %s: %s",
            SERVER_CONFIGURATION_FILE,
            exc,
        )
        return {}
    database_config = data.get("database", {})
    return database_config if isinstance(database_config, dict) else {}


###############################################################################
if __name__ == "__main__":
    start = time.perf_counter()
    logger.info("Starting database initialization")
    logger.info("Current database configuration: %s", json.dumps(load_database_config()))
    initialize_database()
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds", elapsed)
