from __future__ import annotations

import urllib.parse
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from XREPORT.server.common.utils.logger import logger
from XREPORT.server.configurations import DatabaseSettings
from XREPORT.server.repositories.database.utils import (
    normalize_postgres_engine,
    normalize_string_columns,
    resolve_conflict_columns,
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
        Base.metadata.create_all(self.engine)

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
                        stmt = stmt.on_conflict_do_nothing(
                            index_elements=unique_cols
                        )
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
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            if not inspector.has_table(table_name):
                logger.warning("Table %s does not exist", table_name)
                return pd.DataFrame()
            data = pd.read_sql_table(table_name, conn)
        if offset:
            data = data.iloc[offset:]
        if limit is not None:
            data = data.head(limit)
        return data.reset_index(drop=True)

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        with self.engine.begin() as conn:
            inspector = inspect(conn)
            if inspector.has_table(table_name):
                conn.execute(sqlalchemy.text(f'DELETE FROM "{table_name}"'))
            df.to_sql(table_name, conn, if_exists="append", index=False)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(table_name)
        self.upsert_dataframe(df, table_cls)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(f'SELECT COUNT(*) FROM "{table_name}"')
            )
            value = result.scalar() or 0
        return int(value)
