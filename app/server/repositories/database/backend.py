from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from server.common.path import DATABASE_FILE_PATH
from server.common.utils.logger import logger
from server.configurations import DatabaseSettings, get_server_settings
from server.repositories.database.postgres import PostgresRepository
from server.repositories.database.sqlite import SQLiteRepository
from server.repositories.schemas import Base


###############################################################################
class DatabaseBackend(Protocol):
    db_path: Path | None
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
        self.settings = get_server_settings().database
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
            sqlite_db_path = DATABASE_FILE_PATH
            sqlite_database_exists = sqlite_db_path.exists()
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
    def db_path(self) -> Path | None:
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


@lru_cache(maxsize=1)
def get_database() -> XREPORTDatabase:
    return XREPORTDatabase()


__all__ = [
    "DatabaseBackend",
    "XREPORTDatabase",
    "get_database",
    "build_postgres_backend",
    "build_sqlite_backend",
]
