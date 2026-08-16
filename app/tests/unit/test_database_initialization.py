from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy

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
        connect_timeout=30,
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

###############################################################################
def test_missing_sqlite_database_is_created_with_schema(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)

    initializer.initialize_database(_sqlite_settings())

    assert database_path.is_file()
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        assert set(Base.metadata.tables).issubset(
            set(sqlalchemy.inspect(engine).get_table_names())
        )
    finally:
        engine.dispose()

###############################################################################
def test_existing_sqlite_database_is_not_reinitialized_or_validated(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "database.db"
    sentinel = b"existing database bytes"
    database_path.write_bytes(sentinel)
    _patch_sqlite_path(monkeypatch, database_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("existing SQLite database was reinitialized")

    monkeypatch.setattr(initializer.Base.metadata, "create_all", fail_if_called)

    initializer.prepare_database_for_startup(_sqlite_settings())
    initializer.initialize_database(_sqlite_settings())

    assert database_path.read_bytes() == sentinel

###############################################################################
def test_postgres_startup_only_verifies_connection(monkeypatch) -> None:
    settings = _postgres_settings()
    calls: list[str] = []

    monkeypatch.setattr(
        initializer,
        "verify_postgres_connection",
        lambda _settings: calls.append("verify"),
    )
    monkeypatch.setattr(
        initializer,
        "initialize_postgres_database",
        lambda _settings: calls.append("initialize"),
    )

    initializer.prepare_database_for_startup(settings)

    assert calls == ["verify"]

###############################################################################
def test_postgres_startup_connection_failure_is_sanitized(monkeypatch) -> None:
    settings = _postgres_settings()
    failure = sqlalchemy.exc.OperationalError(
        "SELECT 1",
        {},
        OSError("connection refused"),
    )
    monkeypatch.setattr(
        initializer,
        "verify_postgres_connection",
        lambda _settings: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(RuntimeError, match="Database startup check failed") as exc_info:
        initializer.prepare_database_for_startup(settings)

    assert "password" not in str(exc_info.value).lower()

###############################################################################
def test_postgres_initialization_is_only_reached_by_manual_entrypoint(monkeypatch) -> None:
    settings = _postgres_settings()
    calls: list[str] = []
    monkeypatch.setattr(
        initializer,
        "initialize_postgres_database",
        lambda _settings: calls.append("initialize"),
    )

    initializer.initialize_database(settings)

    assert calls == ["initialize"]
