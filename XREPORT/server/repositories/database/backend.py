from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd

from XREPORT.server.common.constants import DATABASE_FILENAME, RESOURCES_PATH
from XREPORT.server.common.utils.logger import logger
from XREPORT.server.configurations import DatabaseSettings, server_settings
from XREPORT.server.repositories.database.postgres import PostgresRepository
from XREPORT.server.repositories.database.sqlite import SQLiteRepository
from XREPORT.server.repositories.schemas import Base


###############################################################################
class DatabaseBackend(Protocol):
    db_path: str | None
    engine: Any

    # -------------------------------------------------------------------------
    def load_from_database(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame: ...

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None: ...

    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None: ...

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int: ...


BackendFactory = Callable[[DatabaseSettings], DatabaseBackend]


# -----------------------------------------------------------------------------
def build_sqlite_backend(settings: DatabaseSettings) -> DatabaseBackend:
    return SQLiteRepository(settings)


# -----------------------------------------------------------------------------
def build_postgres_backend(settings: DatabaseSettings) -> DatabaseBackend:
    return PostgresRepository(settings)


BACKEND_FACTORIES: dict[str, BackendFactory] = {
    "sqlite": build_sqlite_backend,
    "postgres": build_postgres_backend,
}


# [DATABASE]
###############################################################################
class XREPORTDatabase:
    def __init__(self) -> None:
        self.settings = server_settings.database
        self.backend = self._build_backend(self.settings.embedded_database)

    # -------------------------------------------------------------------------
    def _build_backend(self, is_embedded: bool) -> DatabaseBackend:
        backend_name = "sqlite" if is_embedded else (self.settings.engine or "postgres")
        normalized_name = backend_name.lower()
        logger.info("Initializing %s database backend", backend_name)
        if normalized_name not in BACKEND_FACTORIES:
            raise ValueError(f"Unsupported database engine: {backend_name}")
        sqlite_db_path: str | None = None
        sqlite_database_exists = True
        if normalized_name == "sqlite":
            sqlite_db_path = os.path.join(RESOURCES_PATH, DATABASE_FILENAME)
            sqlite_database_exists = os.path.exists(sqlite_db_path)
        factory = BACKEND_FACTORIES[normalized_name]
        backend = factory(self.settings)
        if normalized_name == "sqlite" and not sqlite_database_exists:
            logger.info(
                "SQLite database file missing at startup (%s). Initializing tables.",
                sqlite_db_path,
            )
            Base.metadata.create_all(backend.engine)
        return backend

    # -------------------------------------------------------------------------
    @property
    def db_path(self) -> str | None:
        return getattr(self.backend, "db_path", None)

    # -------------------------------------------------------------------------
    def load_from_database(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        return self.backend.load_from_database(table_name, limit=limit, offset=offset)

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        self.backend.save_into_database(df, table_name)

    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        self.backend.upsert_into_database(df, table_name)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        return self.backend.count_rows(table_name)


database = XREPORTDatabase()


__all__ = [
    "DatabaseBackend",
    "XREPORTDatabase",
    "database",
    "build_postgres_backend",
    "build_sqlite_backend",
]
