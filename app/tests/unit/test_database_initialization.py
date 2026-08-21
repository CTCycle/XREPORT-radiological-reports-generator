from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import sqlalchemy
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext

from server.common.constants import (
    INFERENCE_REPORTS_TABLE,
    INFERENCE_RUNS_TABLE,
    TABLE_REQUIRED_COLUMNS,
)
import server.repositories.database.engine as database_engine
import server.repositories.database.initializer as initializer
from server.configurations.settings import DatabaseSettings
from server.repositories.database.utils import UPSERT_CONFLICT_COLUMNS
from server.repositories.schemas import Base


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


def _patch_sqlite_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(initializer, "DATABASE_FILE_PATH", path)
    monkeypatch.setattr(database_engine, "DATABASE_FILE_PATH", path)


def _alembic_config(path: Path) -> Config:
    config = Config(str(initializer.MIGRATIONS_DIR.parent / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path.as_posix()}")
    return config


def _create_revision(path: Path, revision: str) -> None:
    command.upgrade(_alembic_config(path), revision)


def _database_tables(path: Path) -> set[str]:
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    try:
        return set(sqlalchemy.inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_current_inference_persistence_contract_matches_orm() -> None:
    assert set(TABLE_REQUIRED_COLUMNS[INFERENCE_RUNS_TABLE]) == {
        "checkpoint_id",
        "provider",
        "model_ref",
        "model_revision",
        "generation_profile",
        "generation_config_json",
        "clinical_context",
        "request_id",
        "status",
        "execution_time_seconds",
        "executed_at",
    }
    assert UPSERT_CONFLICT_COLUMNS[INFERENCE_REPORTS_TABLE] == (
        "inference_run_id",
        "input_image_name_key",
    )


def test_empty_sqlite_database_is_migrated_to_head(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)

    initializer.initialize_database(_sqlite_settings())

    assert database_path.is_file()
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        tables = set(sqlalchemy.inspect(engine).get_table_names())
        assert set(Base.metadata.tables).issubset(tables)
        assert "schema_metadata" not in tables
        with engine.connect() as connection:
            version = connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
        assert version == initializer.HEAD_REVISION
    finally:
        engine.dispose()


def test_repeated_sqlite_initialization_is_idempotent(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)

    initializer.initialize_database(_sqlite_settings())
    initializer.prepare_database_for_startup(_sqlite_settings())
    initializer.initialize_database(_sqlite_settings())

    assert _database_tables(database_path) == set(Base.metadata.tables) | {
        initializer.ALEMBIC_VERSION_TABLE
    }


@pytest.mark.parametrize("drop_marker", [False, True])
def test_known_unversioned_schema_is_adopted_without_data_loss(
    tmp_path,
    monkeypatch,
    drop_marker: bool,
) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    _create_revision(database_path, initializer.BASELINE_REVISION)
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO datasets (name, name_key, created_at) "
                "VALUES ('Legacy', 'legacy', '2026-01-01 00:00:00')"
            )
            connection.exec_driver_sql("DROP TABLE alembic_version")
            if drop_marker:
                connection.exec_driver_sql("DROP TABLE schema_metadata")
    finally:
        engine.dispose()

    initializer.initialize_database(_sqlite_settings())

    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        inspector = sqlalchemy.inspect(engine)
        with engine.connect() as connection:
            assert connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one() == initializer.HEAD_REVISION
            assert connection.exec_driver_sql(
                "SELECT name FROM datasets"
            ).scalar_one() == "Legacy"
        assert not inspector.has_table("schema_metadata")
    finally:
        engine.dispose()


def test_database_one_revision_behind_is_upgraded(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    _create_revision(database_path, initializer.BASELINE_REVISION)

    initializer.prepare_database_for_startup(_sqlite_settings())

    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one() == initializer.HEAD_REVISION
    finally:
        engine.dispose()


def test_partial_unversioned_schema_fails_without_stamping(tmp_path, monkeypatch) -> None:
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


def test_unexpected_unversioned_object_fails_without_stamping(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    _create_revision(database_path, initializer.BASELINE_REVISION)
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE unexpected_object (id INTEGER)")
            connection.exec_driver_sql("DROP TABLE alembic_version")
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="unexpected tables"):
        initializer.prepare_database_for_startup(_sqlite_settings())

    assert "alembic_version" not in _database_tables(database_path)


def test_invalid_legacy_marker_fails_without_stamping(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    _create_revision(database_path, initializer.BASELINE_REVISION)
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE schema_metadata SET schema_version = 2 WHERE schema_name = 'xreport'"
            )
            connection.exec_driver_sql("DROP TABLE alembic_version")
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="schema_metadata marker"):
        initializer.prepare_database_for_startup(_sqlite_settings())

    assert "alembic_version" not in _database_tables(database_path)


@pytest.mark.parametrize(
    "version_rows, message",
    [
        (["unknown-revision"], "not present"),
        ([initializer.BASELINE_REVISION, initializer.HEAD_REVISION], "multiple"),
    ],
)
def test_invalid_alembic_version_state_fails(
    tmp_path,
    monkeypatch,
    version_rows: list[str],
    message: str,
) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"
            )
            for revision in version_rows:
                connection.exec_driver_sql(
                    "INSERT INTO alembic_version (version_num) VALUES (?)",
                    (revision,),
                )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match=message):
        initializer.prepare_database_for_startup(_sqlite_settings())


def test_failed_migration_rolls_back_the_shared_transaction(
    tmp_path,
    monkeypatch,
) -> None:
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


def test_migrated_schema_has_no_autogenerate_drift(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    initializer.initialize_database(_sqlite_settings())
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection,
                opts={
                    "compare_type": True,
                    "include_object": lambda _object, name, object_type, *_args: not (
                        object_type == "table" and name == "alembic_version"
                    ),
                },
            )
            assert compare_metadata(context, Base.metadata) == []
    finally:
        engine.dispose()


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


def test_unexpected_head_table_blocks_startup(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    initializer.initialize_database(_sqlite_settings())
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE unexpected_object (id INTEGER)")
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="differs from the current ORM schema"):
        initializer.prepare_database_for_startup(_sqlite_settings())


def test_concurrent_sqlite_initializers_are_serialized(tmp_path, monkeypatch) -> None:
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


def test_postgres_startup_uses_the_same_migration_coordinator(monkeypatch) -> None:
    settings = _postgres_settings()
    calls: list[str] = []
    monkeypatch.setattr(
        initializer,
        "initialize_postgres_database",
        lambda _settings: calls.append("initialize") or "xreport-test",
    )

    initializer.prepare_database_for_startup(settings)

    assert calls == ["initialize"]


def test_postgres_startup_failure_is_sanitized(monkeypatch) -> None:
    settings = _postgres_settings()
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
        initializer.prepare_database_for_startup(settings)

    assert "secret" not in str(exc_info.value).lower()


def test_error_sanitizer_redacts_password_assignments() -> None:
    sanitized = initializer._sanitize_error("password=secret; url=postgresql://user:secret@host/db")

    assert "secret" not in sanitized
    assert sanitized.count("<redacted>") == 2


def test_postgres_initialization_is_reached_by_manual_entrypoint(monkeypatch) -> None:
    settings = _postgres_settings()
    calls: list[str] = []
    monkeypatch.setattr(
        initializer,
        "initialize_postgres_database",
        lambda _settings: calls.append("initialize") or "xreport-test",
    )

    initializer.initialize_database(settings)

    assert calls == ["initialize"]
