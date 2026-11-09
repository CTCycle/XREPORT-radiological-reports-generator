from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from xreport_shared.configurations import get_database_settings
from xreport_shared.repositories.database.engine import (
    Database,
    DatabaseConnectionSnapshot,
)


###############################################################################
@dataclass(frozen=True)
class DatabaseActivation:
    database: Database
    previous_engine: DatabaseConnectionSnapshot | None
    created_facade: bool


###############################################################################
class DatabaseRegistry:
    """Holds the process-wide database facade and guards its activation."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._database: Database | None = None
        self._lock = RLock()

    # -------------------------------------------------------------------------
    def get(self) -> Database:
        with self._lock:
            if self._database is None:
                self._database = Database(get_database_settings())
            return self._database

    # -------------------------------------------------------------------------
    def get_if_initialized(self) -> Database | None:
        with self._lock:
            return self._database

    # -------------------------------------------------------------------------
    def activate(self, candidate: Database) -> DatabaseActivation:
        """Adopt a prepared candidate without replacing cached repository references."""
        with self._lock:
            if self._database is None:
                self._database = candidate
                return DatabaseActivation(
                    database=candidate,
                    previous_engine=None,
                    created_facade=True,
                )
            previous = self._database.replace_connection(candidate)
            return DatabaseActivation(
                database=self._database,
                previous_engine=previous,
                created_facade=False,
            )

    # -------------------------------------------------------------------------
    def rollback(self, activation: DatabaseActivation) -> None:
        with self._lock:
            if activation.created_facade:
                if self._database is activation.database:
                    activation.database.engine.dispose()
                    self._database = None
                return
            if activation.previous_engine is not None:
                activation.database.restore_connection(activation.previous_engine)

    # -------------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            if self._database is not None:
                self._database.engine.dispose()
                self._database = None


database_registry = DatabaseRegistry()
get_database = database_registry.get
get_database_if_initialized = database_registry.get_if_initialized
activate_database = database_registry.activate
rollback_database_activation = database_registry.rollback
reset_database = database_registry.reset


###############################################################################
def finalize_database_activation(activation: DatabaseActivation) -> None:
    if activation.previous_engine is not None:
        Database.dispose_snapshot(activation.previous_engine)


__all__ = [
    "DatabaseActivation",
    "activate_database",
    "finalize_database_activation",
    "get_database",
    "get_database_if_initialized",
    "reset_database",
    "rollback_database_activation",
]
