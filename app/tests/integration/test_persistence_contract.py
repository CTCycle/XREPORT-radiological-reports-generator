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

import xreport_shared.repositories.database.initializer as initializer
from xreport_shared.configurations import DatabaseSettings
from xreport_shared.repositories.database.engine import Database
from xreport_shared.repositories.database.schema import ApplicationSchema, application_schema


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
def _schema_revision(database: Database, schema: ApplicationSchema) -> str:
    with database.engine.connect() as connection:
        return connection.execute(
            text(f"SELECT version_num FROM {schema.version_table}")
        ).scalar_one()


###############################################################################
@pytest.mark.parametrize("profile", ["xreport", "studio"])
def test_postgresql_migrations_and_restart_persistence(
    postgres_settings: DatabaseSettings, profile: str
) -> None:
    schema = application_schema(profile)
    initializer.initialize_postgres_database(postgres_settings, schema)

    database = Database(postgres_settings)
    try:
        assert set(inspect(database.engine).get_table_names()) == set(
            schema.metadata.tables
        ) | {schema.version_table}
        assert _schema_revision(database, schema) == initializer.schema_head(schema)

        with database.transaction() as session:
            settings_row = session.get(schema.settings_model, 1)
            assert settings_row is not None
            settings_row.global_seed = 2_026_092_301
    finally:
        database.engine.dispose()

    # A second startup is the persistence boundary: migrations must remain
    # current while previously committed application data stays intact.
    initializer.initialize_postgres_database(postgres_settings, schema)

    restarted_database = Database(postgres_settings)
    try:
        with restarted_database.transaction() as session:
            settings_row = session.get(schema.settings_model, 1)
            assert settings_row is not None
            assert settings_row.global_seed == 2_026_092_301
        assert _schema_revision(restarted_database, schema) == initializer.schema_head(
            schema
        )
    finally:
        restarted_database.engine.dispose()


###############################################################################
@pytest.mark.parametrize(("profile", "owner"), [("xreport", "studio"), ("studio", "xreport")])
def test_postgresql_database_of_the_other_application_is_rejected(
    postgres_settings: DatabaseSettings, profile: str, owner: str
) -> None:
    owner_schema = application_schema(owner)
    initializer.initialize_postgres_database(postgres_settings, owner_schema)

    with pytest.raises(RuntimeError, match=f"belongs to {owner_schema.title}"):
        initializer.prepare_database_for_startup(
            postgres_settings, application_schema(profile)
        )

    database = Database(postgres_settings)
    try:
        assert set(inspect(database.engine).get_table_names()) == set(
            owner_schema.metadata.tables
        ) | {owner_schema.version_table}
    finally:
        database.engine.dispose()


###############################################################################
def test_postgresql_concurrent_initializers_use_advisory_locks(
    postgres_settings: DatabaseSettings,
) -> None:
    schema = application_schema()
    start = Barrier(2)

    def initialize_together() -> str:
        start.wait(timeout=10)
        return initializer.initialize_postgres_database(postgres_settings, schema)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(initialize_together) for _ in range(2)]
        assert [future.result(timeout=30) for future in futures] == [
            postgres_settings.database_name,
            postgres_settings.database_name,
        ]

    database = Database(postgres_settings)
    try:
        assert _schema_revision(database, schema) == initializer.schema_head(schema)
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
