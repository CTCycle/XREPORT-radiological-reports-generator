from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol

import pandas as pd
from sqlalchemy.orm import declarative_base

from APP.server.utils.configurations import DatabaseSettings, server_settings
from APP.server.utils.logger import logger
from APP.server.database.postgres import PostgresRepository
from APP.server.database.sqlite import SQLiteRepository

Base = declarative_base()


###############################################################################
class DatabaseBackend(Protocol):
    db_path: str | None  
    engine: Any    

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame: ...

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None: ...

    # -------------------------------------------------------------------------
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
class APPDatabase:
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
        factory = BACKEND_FACTORIES[normalized_name]
        return factory(self.settings)
    
    # -------------------------------------------------------------------------
    @property
    def db_path(self) -> str | None:
        return getattr(self.backend, "db_path", None)

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        return self.backend.load_from_database(table_name)

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        self.backend.save_into_database(df, table_name)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        self.backend.upsert_into_database(df, table_name)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        return self.backend.count_rows(table_name)


database = APPDatabase()
