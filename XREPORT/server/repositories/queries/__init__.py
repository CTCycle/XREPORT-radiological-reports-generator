from __future__ import annotations

from XREPORT.server.repositories.queries.postgres import PostgresRepository
from XREPORT.server.repositories.queries.sqlite import SQLiteRepository

__all__ = ["SQLiteRepository", "PostgresRepository"]
