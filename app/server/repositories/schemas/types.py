from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.types import DateTime, JSON, TypeDecorator

###############################################################################
class JSONSequence(TypeDecorator):
    """
    SQLAlchemy TypeDecorator that stores data as JSON.

    Implementation:
    - PostgreSQL: Uses JSONB (via sqlalchemy.types.JSON)
    - SQLite: Uses JSON (via sqlalchemy.types.JSON, stored as TEXT/JSON)

    It relies on SQLAlchemy's built-in JSON support which handles the dialect-specific
    storage implementation automatically.
    """

    impl = JSON
    cache_ok = True

    # -------------------------------------------------------------------------
    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        return value

    # -------------------------------------------------------------------------
    def process_result_value(self, value: Any, dialect: Any) -> Any:
        return value


class UTCDateTime(TypeDecorator):
    """Store and return timezone-aware UTC datetimes on every backend."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("UTCDateTime values must be timezone-aware")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
