from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from xreport_shared.configurations import get_database_settings
from xreport_shared.repositories.database.engine import Database, DatabaseConnectionSnapshot

###############################################################################
@dataclass(frozen=True)
class DatabaseActivation:
    database: Database
    previous_engine: DatabaseConnectionSnapshot | None
    created_facade: bool


_database: Database | None = None
_database_lock = RLock()

###############################################################################
def get_database() -> Database:
    global _database
    with _database_lock:
        if _database is None:
            _database = Database(get_database_settings())
        return _database

###############################################################################
def get_database_if_initialized() -> Database | None:
    with _database_lock:
        return _database

###############################################################################
def activate_database(candidate: Database) -> DatabaseActivation:
    """Adopt a prepared candidate without replacing cached repository references."""
    global _database
    with _database_lock:
        if _database is None:
            _database = candidate
            return DatabaseActivation(
                database=candidate,
                previous_engine=None,
                created_facade=True,
            )
        previous = _database.replace_connection(candidate)
        return DatabaseActivation(
            database=_database,
            previous_engine=previous,
            created_facade=False,
        )

###############################################################################
def rollback_database_activation(activation: DatabaseActivation) -> None:
    global _database
    with _database_lock:
        if activation.created_facade:
            if _database is activation.database:
                activation.database.engine.dispose()
                _database = None
            return
        if activation.previous_engine is not None:
            activation.database.restore_connection(activation.previous_engine)

###############################################################################
def finalize_database_activation(activation: DatabaseActivation) -> None:
    if activation.previous_engine is not None:
        Database.dispose_snapshot(activation.previous_engine)

###############################################################################
def reset_database() -> None:
    global _database
    with _database_lock:
        if _database is not None:
            _database.engine.dispose()
            _database = None


__all__ = [
    "DatabaseActivation",
    "activate_database",
    "finalize_database_activation",
    "get_database",
    "get_database_if_initialized",
    "reset_database",
    "rollback_database_activation",
]
