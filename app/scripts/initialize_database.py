from __future__ import annotations

import argparse
import os
import time


###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize an XREPORT database")
    parser.add_argument("--app", choices=("xreport", "studio"), default="xreport")
    arguments = parser.parse_args()
    # Data roots are resolved when the server package is imported, so the
    # profile must be exported first; an explicit environment value must agree.
    configured = os.environ.setdefault("XREPORT_APP", arguments.app).strip().lower()
    if configured != arguments.app:
        raise SystemExit(
            f"XREPORT_APP must be '{arguments.app}' to initialize the {arguments.app} database"
        )

    # Importing the application package registers the schema it owns.
    import importlib

    importlib.import_module(
        "xreport_studio" if arguments.app == "studio" else "xreport_inference"
    )
    from xreport_shared.common.utils.logger import logger
    from xreport_shared.configurations import get_database_settings
    from xreport_shared.repositories.database.initializer import initialize_database

    start = time.perf_counter()
    database_settings = get_database_settings()
    logger.info(
        "Starting %s database initialization for %s",
        arguments.app,
        database_settings.backend,
    )
    initialize_database(database_settings)
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds", elapsed)
