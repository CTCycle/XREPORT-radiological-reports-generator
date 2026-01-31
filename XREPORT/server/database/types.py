from __future__ import annotations

from typing import Any

from sqlalchemy.types import TypeDecorator, JSON


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

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        return value

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        return value


###############################################################################
class IntSequence(JSONSequence):
    """
    Specialized JSONSequence for integer lists.
    Functionally identical to JSONSequence but serves as semantic documentation.
    """

    pass
