from __future__ import annotations

import urllib.parse
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import delete, func, inspect, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from XREPORT.server.common.utils.logger import logger
from XREPORT.server.configurations import DatabaseSettings
from XREPORT.server.repositories.database.utils import (
    normalize_postgres_engine,
    normalize_string_columns,
    resolve_conflict_columns,
    validate_table_name,
    validate_unique_key_values,
)
from XREPORT.server.repositories.schemas import Base


###############################################################################
class PostgresRepository:
    def __init__(self, settings: DatabaseSettings) -> None:
        if not settings.host:
            raise ValueError("Database host must be provided for external database.")
        if not settings.database_name:
            raise ValueError("Database name must be provided for external database.")
        if not settings.username:
            raise ValueError(
                "Database username must be provided for external database."
            )

        port = settings.port or 5432
        engine_name = normalize_postgres_engine(settings.engine)
        password = settings.password or ""
        connect_args: dict[str, Any] = {"connect_timeout": settings.connect_timeout}
        if settings.ssl:
            connect_args["sslmode"] = "require"
            if settings.ssl_ca:
                connect_args["sslrootcert"] = settings.ssl_ca

        safe_username = urllib.parse.quote_plus(settings.username)
        safe_password = urllib.parse.quote_plus(password)
        self.db_path: str | None = None
        self.engine: Engine = sqlalchemy.create_engine(
            f"{engine_name}://{safe_username}:{safe_password}@{settings.host}:{port}/{settings.database_name}",
            echo=False,
            future=True,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        self.session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = settings.insert_batch_size

    # -------------------------------------------------------------------------
    def get_table_class(self, table_name: str) -> Any:
        for cls in Base.__subclasses__():
            if getattr(cls, "__tablename__", None) == table_name:
                return cls
        raise ValueError(f"No table class found for name {table_name}")

    # -------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls) -> None:
        table = table_cls.__table__
        table_name = str(getattr(table, "name", ""))
        session = self.session()
        try:
            payload_columns = [str(column) for column in df.columns]
            unique_cols, missing_cols = resolve_conflict_columns(
                table_name, payload_columns
            )
            if missing_cols:
                raise ValueError(
                    f"Missing conflict columns for {table_name}: "
                    f"{', '.join(missing_cols)}"
                )
            normalized = normalize_string_columns(df)
            records = normalized.to_dict(orient="records")
            validate_unique_key_values(records, unique_cols, table_name)
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i : i + self.insert_batch_size]
                if not batch:
                    continue
                stmt = insert(table).values(batch)
                if unique_cols:
                    update_cols = {
                        col: getattr(stmt.excluded, col)  # type: ignore[attr-defined]
                        for col in batch[0]
                        if col not in unique_cols
                    }
                    if update_cols:
                        stmt = stmt.on_conflict_do_update(
                            index_elements=unique_cols, set_=update_cols
                        )
                    else:
                        stmt = stmt.on_conflict_do_nothing(index_elements=unique_cols)
                session.execute(stmt)
                session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def load_from_database(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        safe_table_name = validate_table_name(table_name)
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            if not inspector.has_table(safe_table_name):
                logger.warning("Table %s does not exist", safe_table_name)
                return pd.DataFrame()

        table_cls = self.get_table_class(safe_table_name)
        primary_keys = [column.name for column in table_cls.__table__.primary_key.columns]
        order_columns = [getattr(table_cls, key) for key in primary_keys]
        stmt = select(table_cls)
        if order_columns:
            stmt = stmt.order_by(*order_columns)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        session = self.session()
        try:
            instances = session.execute(stmt).scalars().all()
        finally:
            session.close()
        if not instances:
            return pd.DataFrame()
        rows = [
            {
                column.name: getattr(instance, column.name)
                for column in table_cls.__table__.columns
            }
            for instance in instances
        ]
        return pd.DataFrame(rows).reset_index(drop=True)

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        safe_table_name = validate_table_name(table_name)
        table_cls = self.get_table_class(safe_table_name)
        normalized = normalize_string_columns(df)
        records = normalized.to_dict(orient="records")

        session = self.session()
        try:
            session.execute(delete(table_cls))
            if records:
                session.add_all([table_cls(**record) for record in records])
            session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        safe_table_name = validate_table_name(table_name)
        table_cls = self.get_table_class(safe_table_name)
        self.upsert_dataframe(df, table_cls)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        safe_table_name = validate_table_name(table_name)
        table_cls = self.get_table_class(safe_table_name)
        session = self.session()
        try:
            value = (
                session.execute(select(func.count()).select_from(table_cls)).scalar() or 0
            )
        finally:
            session.close()
        return int(value)
