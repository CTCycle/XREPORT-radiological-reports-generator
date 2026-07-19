from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from server.common.constants import CHECKPOINTS_TABLE, DATASETS_TABLE, TABLE_REQUIRED_COLUMNS
from server.common.path import CHECKPOINTS_DIR
from server.common.utils.logger import logger
from server.common.utils.security import validate_checkpoint_name
from server.repositories.database import Database, get_database
from server.repositories.database.utils import validate_sql_identifier, validate_table_name
from server.repositories.schemas import Checkpoint, Dataset
from server.repositories.schemas.normalization import normalize_key

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
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str) and value.strip():
            parsed = datetime.fromisoformat(value.strip())
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
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
        missing = [column for column in required_columns if column not in dataset.columns]
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
            self.validate_required_columns(dataset, required_columns, table_name, "save")
        self.database.save_into_database(dataset, table_name)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        if dataset.empty:
            logger.debug("Skipping upsert for %s: dataset is empty", table_name)
            return
        required_columns = TABLE_REQUIRED_COLUMNS.get(table_name)
        if required_columns:
            self.validate_required_columns(dataset, required_columns, table_name, "upsert")
        self.database.upsert_into_database(dataset, table_name)

    # -------------------------------------------------------------------------
    def _session(self) -> Session:
        return self.database.session()

    # -------------------------------------------------------------------------
    def _get_table_class(self, table_name: str) -> Any:
        return self.database.get_table_class(validate_table_name(table_name))

    # -------------------------------------------------------------------------
    def _get_dataset_id(self, dataset_name: str) -> int | None:
        with self.database.read_session() as session:
            dataset_id = session.execute(
                select(Dataset.dataset_id).where(
                    Dataset.name_key == normalize_key(dataset_name)
                )
            ).scalar_one_or_none()
        return int(dataset_id) if dataset_id is not None else None

    # -------------------------------------------------------------------------
    def _ensure_dataset(self, dataset_name: str) -> int:
        normalized_name = str(dataset_name or "").strip()
        if not normalized_name:
            raise ValueError("Dataset name cannot be empty")
        existing_id = self._get_dataset_id(normalized_name)
        if existing_id is not None:
            return existing_id
        self.upsert_table(
            pd.DataFrame(
                [
                    {
                        "name": normalized_name,
                        "name_key": normalize_key(normalized_name),
                        "created_at": self._now_utc(),
                    }
                ]
            ),
            DATASETS_TABLE,
        )
        created_id = self._get_dataset_id(normalized_name)
        if created_id is None:
            raise RuntimeError(f"Failed to create dataset: {normalized_name}")
        return created_id

    # -------------------------------------------------------------------------
    def _ensure_checkpoint(self, checkpoint: str) -> int:
        checkpoint_name = validate_checkpoint_name(checkpoint)
        with self.database.read_session() as session:
            checkpoint_id = session.execute(
                select(Checkpoint.checkpoint_id).where(
                    Checkpoint.name_key == normalize_key(checkpoint_name)
                )
            ).scalar_one_or_none()
        if checkpoint_id is not None:
            return int(checkpoint_id)
        self.upsert_table(
            pd.DataFrame(
                [
                    {
                        "name": checkpoint_name,
                        "name_key": normalize_key(checkpoint_name),
                        "path": str(CHECKPOINTS_DIR / checkpoint_name),
                        "created_at": self._now_utc(),
                        "last_seen_at": self._now_utc(),
                    }
                ]
            ),
            CHECKPOINTS_TABLE,
        )
        with self.database.read_session() as session:
            checkpoint_id = session.execute(
                select(Checkpoint.checkpoint_id).where(
                    Checkpoint.name_key == normalize_key(checkpoint_name)
                )
            ).scalar_one_or_none()
        if checkpoint_id is None:
            raise RuntimeError(f"Failed to create checkpoint row: {checkpoint_name}")
        return int(checkpoint_id)

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
