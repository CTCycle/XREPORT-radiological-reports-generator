from __future__ import annotations

from functools import lru_cache
from server.configurations import get_server_settings
from server.repositories.database.engine import Database


# [DATABASE]

###############################################################################
@lru_cache(maxsize=1)
def get_database() -> Database:
    return Database(get_server_settings().database)


__all__ = [
    "get_database",
]
