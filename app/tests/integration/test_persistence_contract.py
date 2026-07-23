from __future__ import annotations

import os

import pytest
from sqlalchemy import inspect, text

from server.configurations import DatabaseSettings
from server.repositories.database.engine import Database
from server.repositories.schemas import Base

###############################################################################
def test_postgresql_schema_contract() -> None:
    if os.getenv("EMBEDDED_DATABASE", "true").lower() != "false":
        pytest.skip("PostgreSQL integration tests require EMBEDDED_DATABASE=false")
    settings = DatabaseSettings(
        backend="postgresql",
        engine="postgresql+psycopg",
        host=os.environ["DATABASE_HOST"],
        port=int(os.environ["DATABASE_PORT"]),
        database_name=os.environ["DATABASE_NAME"],
        username=os.environ["DATABASE_USERNAME"],
        password=os.environ["DATABASE_PASSWORD"],
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
        inference_columns = {
            column["name"]: column
            for column in inspect(database.engine).get_columns("inference_runs")
        }
        assert {
            "provider",
            "model_ref",
            "model_revision",
            "generation_profile",
            "generation_config_json",
            "clinical_context",
            "request_id",
            "status",
            "execution_time_seconds",
        } <= set(inference_columns)
        assert inference_columns["checkpoint_id"]["nullable"] is True
        with database.transaction() as session:
            session.execute(text("SELECT 1"))
    finally:
        Base.metadata.drop_all(database.engine)
        database.engine.dispose()
