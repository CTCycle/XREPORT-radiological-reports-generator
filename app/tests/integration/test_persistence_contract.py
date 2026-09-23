from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from uuid import uuid4

import pytest
import sqlalchemy
from sqlalchemy import inspect, text

import server.repositories.database.initializer as initializer
from server.configurations import DatabaseSettings
from server.repositories.database.engine import Database
from server.repositories.schemas.models import ApplicationSettingsRecord

###############################################################################
def _postgres_settings() -> DatabaseSettings:
    if os.getenv("XREPORT_RUN_POSTGRES_INTEGRATION", "false").lower() != "true":
        pytest.skip("PostgreSQL integration test requires an explicit test database")

    return DatabaseSettings(
        backend="postgresql",
        engine="postgresql+psycopg",
        host=os.environ["XREPORT_TEST_DATABASE_HOST"],
        port=int(os.environ["XREPORT_TEST_DATABASE_PORT"]),
        database_name=os.environ["XREPORT_TEST_DATABASE_NAME"],
        username=os.environ["XREPORT_TEST_DATABASE_USERNAME"],
        password=os.environ["XREPORT_TEST_DATABASE_PASSWORD"],
        ssl=False,
        ssl_ca=None,
        connect_timeout=10,
        insert_batch_size=100,
    )

###############################################################################
def _drop_postgres_database(settings: DatabaseSettings) -> None:
    database_name = settings.database_name
    assert database_name is not None
    admin_engine = sqlalchemy.create_engine(
        initializer.build_postgres_url(settings, "postgres"),
        connect_args=initializer.build_postgres_connect_args(settings),
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )
    try:
        with admin_engine.connect() as connection:
            database_exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": database_name},
            ).scalar()
            if database_exists:
                connection.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) "
                        "FROM pg_stat_activity "
                        "WHERE datname = :name AND pid <> pg_backend_pid()"
                    ),
                    {"name": database_name},
                )
                safe_name = database_name.replace('"', '""')
                connection.execute(text(f'DROP DATABASE "{safe_name}"'))
    finally:
        admin_engine.dispose()

###############################################################################
@pytest.fixture
def postgres_settings() -> Iterator[DatabaseSettings]:
    base_settings = _postgres_settings()
    database_name = f"xreport_s14_{uuid4().hex[:12]}"
    settings = replace(base_settings, database_name=database_name)
    try:
        yield settings
    finally:
        _drop_postgres_database(settings)

###############################################################################
def test_postgresql_migrations_and_restart_persistence(
    postgres_settings: DatabaseSettings,
) -> None:
    initializer.initialize_postgres_database(postgres_settings)

    database = Database(postgres_settings)
    try:
        tables = set(inspect(database.engine).get_table_names())
        assert {
            "application_settings",
            "datasets",
            "dataset_versions",
            "dataset_records",
            "alembic_version",
        } <= tables

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

        with database.engine.connect() as connection:
            assert (
                connection.execute(text("SELECT version_num FROM alembic_version"))
                .scalar_one()
                == initializer.HEAD_REVISION
            )
            assert connection.execute(text("SELECT 1")).scalar_one() == 1

        with database.transaction() as session:
            settings_row = session.get(ApplicationSettingsRecord, 1)
            assert settings_row is not None
            settings_row.global_seed = 2_026_092_301
    finally:
        database.engine.dispose()

    # A second startup is the persistence boundary: migrations must remain
    # current while previously committed application data stays intact.
    initializer.initialize_postgres_database(postgres_settings)

    restarted_database = Database(postgres_settings)
    try:
        with restarted_database.transaction() as session:
            settings_row = session.get(ApplicationSettingsRecord, 1)
            assert settings_row is not None
            assert settings_row.global_seed == 2_026_092_301
        with restarted_database.engine.connect() as connection:
            assert (
                connection.execute(text("SELECT version_num FROM alembic_version"))
                .scalar_one()
                == initializer.HEAD_REVISION
            )
    finally:
        restarted_database.engine.dispose()

###############################################################################
def test_postgresql_concurrent_initializers_use_advisory_locks(
    postgres_settings: DatabaseSettings,
) -> None:
    start = Barrier(2)

    def initialize_together() -> str:
        start.wait(timeout=10)
        return initializer.initialize_postgres_database(postgres_settings)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(initialize_together) for _ in range(2)]
        assert [future.result(timeout=30) for future in futures] == [
            postgres_settings.database_name,
            postgres_settings.database_name,
        ]

    database = Database(postgres_settings)
    try:
        with database.engine.connect() as connection:
            assert (
                connection.execute(text("SELECT version_num FROM alembic_version"))
                .scalar_one()
                == initializer.HEAD_REVISION
            )
        assert "application_settings" in set(inspect(database.engine).get_table_names())
    finally:
        database.engine.dispose()

###############################################################################
def test_postgresql_connection_failure_does_not_leak_credentials(
    postgres_settings: DatabaseSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "s14-integration-secret"
    unreachable_settings = replace(
        postgres_settings,
        host="127.0.0.1",
        port=1,
        password=secret,
        connect_timeout=2,
    )
    caplog.set_level(logging.ERROR)

    with pytest.raises(
        RuntimeError, match="Database startup migration failed"
    ) as exc_info:
        initializer.prepare_database_for_startup(unreachable_settings)

    assert secret not in str(exc_info.value)
    assert secret not in caplog.text
    assert "CREATEDB privilege" in str(exc_info.value)
