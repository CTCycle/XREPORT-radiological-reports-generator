from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import sqlalchemy
from alembic import command
from alembic.config import Config

import server.repositories.database.engine as database_engine
import server.repositories.database.initializer as initializer
from server.configurations.settings import DatabaseSettings
from server.repositories.schemas import Base

###############################################################################
def _sqlite_settings() -> DatabaseSettings:
    return DatabaseSettings(
        backend="sqlite",
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=3,
        insert_batch_size=1000,
    )

###############################################################################
def _postgres_settings() -> DatabaseSettings:
    return DatabaseSettings(
        backend="postgresql",
        engine="postgresql+psycopg",
        host="127.0.0.1",
        port=5432,
        database_name="xreport-test",
        username="xreport",
        password="password",
        ssl=False,
        ssl_ca=None,
        connect_timeout=1,
        insert_batch_size=1000,
    )

###############################################################################
def _patch_sqlite_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(initializer, "DATABASE_FILE_PATH", path)
    monkeypatch.setattr(database_engine, "DATABASE_FILE_PATH", path)

###############################################################################
def _alembic_config(path: Path) -> Config:
    config = Config(str(initializer.MIGRATIONS_DIR.parent / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path.as_posix()}")
    return config

###############################################################################
def _database_tables(path: Path) -> set[str]:
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    try:
        return set(sqlalchemy.inspect(engine).get_table_names())
    finally:
        engine.dispose()

###############################################################################
def test_sqlite_startup_migrates_to_head_and_is_repeatable(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)

    initializer.initialize_database(_sqlite_settings())
    initializer.prepare_database_for_startup(_sqlite_settings())

    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        tables = set(sqlalchemy.inspect(engine).get_table_names())
        assert set(Base.metadata.tables).issubset(tables)
        with engine.connect() as connection:
            assert connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one() == initializer.HEAD_REVISION
    finally:
        engine.dispose()

###############################################################################
def test_known_unversioned_schema_is_adopted_without_data_loss(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    command.upgrade(_alembic_config(database_path), initializer.BASELINE_REVISION)
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO datasets (name, name_key, created_at) "
                "VALUES ('Legacy', 'legacy', '2026-01-01 00:00:00')"
            )
            connection.exec_driver_sql("DROP TABLE alembic_version")
    finally:
        engine.dispose()

    initializer.initialize_database(_sqlite_settings())

    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one() == initializer.HEAD_REVISION
            assert connection.exec_driver_sql("SELECT name FROM datasets").scalar_one() == "Legacy"
    finally:
        engine.dispose()

###############################################################################
def test_unknown_partial_schema_is_rejected_without_stamping(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE datasets (dataset_id INTEGER PRIMARY KEY, name TEXT)"
            )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="known XREPORT v1 schema"):
        initializer.prepare_database_for_startup(_sqlite_settings())

    assert "alembic_version" not in _database_tables(database_path)

###############################################################################
def test_failed_migration_rolls_back_database_changes(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)

    def fail_upgrade(config: Config, _revision: str) -> None:
        connection = config.attributes["connection"]
        connection.exec_driver_sql("CREATE TABLE transient_failure (id INTEGER)")
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(initializer.command, "upgrade", fail_upgrade)
    with pytest.raises(RuntimeError, match="Unexpected error"):
        initializer.initialize_database(_sqlite_settings())

    assert "transient_failure" not in _database_tables(database_path)
    assert "alembic_version" not in _database_tables(database_path)

###############################################################################
def test_head_schema_drift_blocks_startup(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    initializer.initialize_database(_sqlite_settings())
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("ALTER TABLE datasets ADD COLUMN unexpected TEXT")
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="differs from the current ORM schema"):
        initializer.prepare_database_for_startup(_sqlite_settings())

###############################################################################
def test_concurrent_sqlite_initialization_is_safe(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    settings = _sqlite_settings()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(initializer.initialize_database, settings) for _ in range(2)]
        for future in futures:
            future.result()

    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one() == initializer.HEAD_REVISION
    finally:
        engine.dispose()

###############################################################################
def test_postgres_startup_failure_does_not_leak_credentials(monkeypatch) -> None:
    failure = sqlalchemy.exc.OperationalError(
        "postgresql://xreport:secret@127.0.0.1/xreport-test",
        {},
        OSError("connection refused"),
    )
    monkeypatch.setattr(
        initializer,
        "initialize_postgres_database",
        lambda _settings: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(RuntimeError, match="Database startup migration failed") as exc_info:
        initializer.prepare_database_for_startup(_postgres_settings())

    assert "secret" not in str(exc_info.value).lower()
