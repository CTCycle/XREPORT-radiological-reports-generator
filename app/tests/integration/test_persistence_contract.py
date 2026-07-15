from __future__ import annotations

import os

import pytest
from sqlalchemy import inspect, text

from server.configurations import DatabaseSettings
from server.repositories.database.engine import Database
from server.repositories.schemas import Base


def test_postgresql_schema_contract() -> None:
    if os.getenv("XREPORT_DB_BACKEND") != "postgresql":
        pytest.fail("PostgreSQL integration tests require XREPORT_DB_BACKEND=postgresql")
    settings = DatabaseSettings(
        backend="postgresql",
        engine="postgresql+psycopg",
        host=os.environ["XREPORT_DB_HOST"],
        port=int(os.environ["XREPORT_DB_PORT"]),
        database_name=os.environ["XREPORT_DB_NAME"],
        username=os.environ["XREPORT_DB_USERNAME"],
        password=os.environ["XREPORT_DB_PASSWORD"],
        ssl=False,
        ssl_ca=None,
        connect_timeout=10,
        insert_batch_size=100,
    )
    database = Database(settings)
    try:
        Base.metadata.create_all(database.engine)
        tables = set(inspect(database.engine).get_table_names())
        assert {"datasets", "dataset_versions", "dataset_records"} <= tables
        with database.transaction() as session:
            session.execute(text("SELECT 1"))
    finally:
        Base.metadata.drop_all(database.engine)
        database.engine.dispose()
