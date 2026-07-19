from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import sqlalchemy
from sqlalchemy import delete, event, func, inspect, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from server.common.path import DATABASE_FILE_PATH
from server.common.utils.logger import logger
from server.configurations import DatabaseSettings
from server.repositories.database.utils import (
    normalize_postgres_engine,
    normalize_string_columns,
    resolve_conflict_columns,
    validate_table_name,
    validate_unique_key_values,
)
from server.repositories.schemas import Base

###############################################################################
class Database:
    """Shared SQLAlchemy database implementation for SQLite and PostgreSQL."""

    # -------------------------------------------------------------------------
    def __init__(self, settings: DatabaseSettings) -> None:
        self.db_path: Path | None = None
        self.insert_batch_size = settings.insert_batch_size
        if settings.backend == "sqlite":
            self.db_path = DATABASE_FILE_PATH
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = sqlalchemy.create_engine(
                f"sqlite:///{self.db_path}", future=True, pool_pre_ping=True
            )
            event.listen(self.engine, "connect", self._configure_sqlite)
        else:
            if not settings.host or not settings.database_name or not settings.username:
                raise ValueError("PostgreSQL host, database, and username are required")
            scheme = normalize_postgres_engine(settings.engine)
            username = sqlalchemy.engine.URL.create(
                scheme,
                username=settings.username,
                password=settings.password or "",
                host=settings.host,
                port=settings.port or 5432,
                database=settings.database_name,
            )
            connect_args: dict[str, Any] = {"connect_timeout": settings.connect_timeout}
            if settings.ssl:
                connect_args["sslmode"] = "require"
                if settings.ssl_ca:
                    connect_args["sslrootcert"] = settings.ssl_ca
            self.engine = sqlalchemy.create_engine(
                username, future=True, pool_pre_ping=True, connect_args=connect_args
            )
        self.session = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)

    # -------------------------------------------------------------------------
    @staticmethod
    def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    @contextmanager
    def read_session(self) -> Iterator[Session]:
        session = self.session()
        try:
            yield session
        finally:
            session.close()

    # -------------------------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[Session]:
        session = self.session()
        try:
            with session.begin():
                yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def get_table_class(self, table_name: str) -> Any:
        for cls in Base.__subclasses__():
            if getattr(cls, "__tablename__", None) == table_name:
                return cls
        raise ValueError(f"No table class found for name {table_name}")

    # -------------------------------------------------------------------------
    def load_from_database(
        self, table_name: str, limit: int | None = None, offset: int | None = None
    ) -> pd.DataFrame:
        safe_name = validate_table_name(table_name)
        with self.engine.connect() as conn:
            if not inspect(conn).has_table(safe_name):
                logger.warning("Table %s does not exist", safe_name)
                return pd.DataFrame()
        table_cls = self.get_table_class(safe_name)
        stmt = select(table_cls).order_by(*table_cls.__table__.primary_key.columns)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        with self.read_session() as session:
            rows = session.execute(stmt).mappings().all()
        return pd.DataFrame([dict(row) for row in rows]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    def _insert_dataframe(self, df: pd.DataFrame, table_cls: Any, *, upsert: bool) -> None:
        if df.empty:
            return
        table_name = str(table_cls.__table__.name)
        records = normalize_string_columns(df).to_dict(orient="records")
        unique_cols, missing_cols = resolve_conflict_columns(table_name, list(df.columns))
        if missing_cols:
            raise ValueError(
                f"Missing conflict columns for {table_name}: {', '.join(missing_cols)}"
            )
        validate_unique_key_values(records, unique_cols, table_name)
        dialect_name = self.engine.dialect.name
        insert = {"sqlite": sqlite_insert, "postgresql": postgres_insert}.get(dialect_name) if upsert else None
        with self.transaction() as session:
            for start in range(0, len(records), self.insert_batch_size):
                batch = records[start : start + self.insert_batch_size]
                if insert is None:
                    session.add_all(table_cls(**record) for record in batch)
                    continue
                stmt = insert(table_cls.__table__).values(batch)
                if not unique_cols:
                    session.execute(stmt)
                    continue
                update_cols = {
                    col: getattr(stmt.excluded, col)
                    for col in batch[0]
                    if col not in unique_cols
                }
                stmt = (
                    stmt.on_conflict_do_update(index_elements=unique_cols, set_=update_cols)
                    if update_cols
                    else stmt.on_conflict_do_nothing(index_elements=unique_cols)
                )
                session.execute(stmt)

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(validate_table_name(table_name))
        records = normalize_string_columns(df).to_dict(orient="records")
        with self.transaction() as session:
            session.execute(delete(table_cls))
            session.add_all(table_cls(**record) for record in records)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(validate_table_name(table_name))
        self._insert_dataframe(df, table_cls, upsert=True)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        table_cls = self.get_table_class(validate_table_name(table_name))
        with self.read_session() as session:
            return int(session.execute(select(func.count()).select_from(table_cls)).scalar() or 0)
