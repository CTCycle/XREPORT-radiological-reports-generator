from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from server.common.constants import TABLE_REQUIRED_COLUMNS
from server.common.utils.logger import logger
from server.repositories.database import Database, get_database
from server.repositories.database.utils import (
    validate_sql_identifier,
    validate_table_name,
)

###############################################################################
class RepositorySupport:
    """Shared database primitives for independent persistence repositories."""

    # -------------------------------------------------------------------------
    def __init__(self, database: Database | None = None) -> None:
        self.database = database or get_database()

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_json(value: Any, default: Any = None) -> Any:
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str) and value.strip():
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return default
            return decoded if isinstance(decoded, (dict, list)) else default
        return default

    # -------------------------------------------------------------------------
    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return (
                value
                if value.tzinfo is not None
                else value.replace(tzinfo=timezone.utc)
            )
        if isinstance(value, str) and value.strip():
            parsed = datetime.fromisoformat(value.strip())
            return (
                parsed
                if parsed.tzinfo is not None
                else parsed.replace(tzinfo=timezone.utc)
            )
        return RepositorySupport._now_utc()

    # -------------------------------------------------------------------------
    @staticmethod
    def _format_datetime(value: Any) -> str | None:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, str):
            return value
        return None

    # -------------------------------------------------------------------------
    def validate_required_columns(
        self,
        dataset: pd.DataFrame,
        required_columns: list[str],
        table_name: str,
        operation: str,
    ) -> None:
        missing = [
            column for column in required_columns if column not in dataset.columns
        ]
        if missing:
            raise ValueError(
                f"Missing required columns for {table_name} {operation}: {', '.join(missing)}"
            )

    # -------------------------------------------------------------------------
    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        if offset is not None and offset < 0:
            raise ValueError("offset must be >= 0")
        return self.database.load_from_database(table_name, limit=limit, offset=offset)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        return self.database.count_rows(table_name)

    # -------------------------------------------------------------------------
    def save_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        if dataset.empty:
            logger.debug("Skipping save for %s: dataset is empty", table_name)
            return
        required_columns = TABLE_REQUIRED_COLUMNS.get(table_name)
        if required_columns:
            self.validate_required_columns(
                dataset, required_columns, table_name, "save"
            )
        self.database.save_into_database(dataset, table_name)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        if dataset.empty:
            logger.debug("Skipping upsert for %s: dataset is empty", table_name)
            return
        required_columns = TABLE_REQUIRED_COLUMNS.get(table_name)
        if required_columns:
            self.validate_required_columns(
                dataset, required_columns, table_name, "upsert"
            )
        self.database.upsert_into_database(dataset, table_name)

    # -------------------------------------------------------------------------
    def _session(self) -> Session:
        return self.database.session()

    # -------------------------------------------------------------------------
    def _get_table_class(self, table_name: str) -> Any:
        return self.database.get_table_class(validate_table_name(table_name))

    # -------------------------------------------------------------------------
    def _delete_by_key(self, table_name: str, column_name: str, value: Any) -> None:
        table_class = self._get_table_class(table_name)
        safe_column_name = validate_sql_identifier(column_name)
        column = getattr(table_class, safe_column_name, None)
        if column is None:
            raise ValueError(
                f"Column {safe_column_name} does not exist on table {table_name}"
            )
        with self.database.transaction() as session:
            session.execute(delete(table_class).where(column == value))
