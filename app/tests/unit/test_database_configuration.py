from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import server.configurations.database_config as database_config
import server.repositories.database.backend as database_backend
import server.repositories.database.initializer as initializer
import server.services.database_settings as database_settings_service
from server.configurations.database_config import (
    DatabaseConfiguration,
    DatabaseConfigurationError,
)
from server.repositories.application_settings import ApplicationSettingsRepository
from server.repositories.database.engine import Database
from server.repositories.schemas import ApplicationSettingsRecord
from server.services.database_settings import DatabaseSwitchBusyError, DatabaseSettingsService

###############################################################################
def _sqlite(path: Path) -> DatabaseConfiguration:
    return DatabaseConfiguration(backend="sqlite", sqlite_path=path)

###############################################################################
def _seed_database(settings, seed: int) -> None:
    initializer.initialize_database(settings)
    database = Database(settings)
    try:
        with database.transaction() as session:
            session.query(ApplicationSettingsRecord).update({"global_seed": seed})
    finally:
        database.engine.dispose()

###############################################################################
def test_sqlite_configuration_requires_an_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="absolute"):
        DatabaseConfiguration(backend="sqlite", sqlite_path=Path("relative.db"))

###############################################################################
def test_postgres_configuration_requires_connection_identity() -> None:
    with pytest.raises(ValidationError, match="host"):
        DatabaseConfiguration(backend="postgresql")

###############################################################################
@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI is the credential store")
def test_database_config_protects_password_and_roundtrips(tmp_path: Path) -> None:
    configuration = DatabaseConfiguration(
        backend="postgresql",
        host="db.example.test",
        database_name="xreport",
        username="xreport",
        password="top-secret-value",
    )
    path = tmp_path / "database-config.json"

    database_config.save_database_configuration(configuration, path=path)
    saved = path.read_text(encoding="utf-8")
    payload = json.loads(saved)

    assert "top-secret-value" not in saved
    assert payload["password_protected"]
    assert "password" not in payload
    assert database_config.read_database_configuration(path=path).password == "top-secret-value"

###############################################################################
def test_invalid_database_config_is_preserved_for_recovery(tmp_path: Path) -> None:
    path = tmp_path / "database-config.json"
    path.write_text('{"schema_version":99,"backend":"sqlite"}\n', encoding="utf-8")
    original = path.read_bytes()

    status = database_config.read_database_configuration_status(path=path)

    assert status.configuration is None
    assert status.error
    assert path.read_bytes() == original

###############################################################################
def test_connection_test_reports_an_unavailable_postgres_server() -> None:
    configuration = DatabaseConfiguration(
        backend="postgresql",
        host="127.0.0.1",
        port=1,
        database_name="unavailable",
        username="xreport",
        connect_timeout=1,
    )

    result = DatabaseSettingsService().test_connection(configuration)

    assert result.success is False
    assert result.connected is False
    assert result.schema_ready is False
    assert result.detail

###############################################################################
def test_sqlite_switch_reloads_active_database_settings_and_can_roll_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_settings = _sqlite(tmp_path / "original.db").to_database_settings()
    target_configuration = _sqlite(tmp_path / "target.db")
    target_settings = target_configuration.to_database_settings()
    _seed_database(original_settings, 11)
    _seed_database(target_settings, 22)

    monkeypatch.setattr(
        database_settings_service,
        "get_job_manager",
        lambda: SimpleNamespace(is_job_running=lambda: False),
    )
    monkeypatch.setattr(
        "server.services.startup_validation.run_startup_validations",
        lambda settings: settings,
    )
    config_path = tmp_path / "database-config.json"
    monkeypatch.setattr(
        database_settings_service,
        "save_database_configuration",
        lambda configuration: database_config.save_database_configuration(
            configuration, path=config_path
        ),
    )

    database_backend.reset_database()
    try:
        database_backend.activate_database(Database(original_settings))
        facade = database_backend.get_database()
        service = DatabaseSettingsService()

        result = service.switch_database(target_configuration)

        assert database_backend.get_database() is facade
        assert result.response.settings.values.global_settings.seed == 22
        assert ApplicationSettingsRepository().get_settings().global_seed == 22
        assert database_config.read_database_configuration(path=config_path) == target_configuration

        def fail_save(_configuration, **_kwargs) -> None:
            raise DatabaseConfigurationError("simulated configuration write failure")

        monkeypatch.setattr(database_settings_service, "save_database_configuration", fail_save)
        with pytest.raises(DatabaseConfigurationError, match="simulated configuration write failure"):
            service.switch_database(_sqlite(tmp_path / "original.db"))

        assert database_backend.get_database() is facade
        assert ApplicationSettingsRepository().get_settings().global_seed == 22
        assert database_config.read_database_configuration(path=config_path) == target_configuration
    finally:
        database_backend.reset_database()

###############################################################################
def test_database_switch_is_blocked_while_a_job_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        database_settings_service,
        "get_job_manager",
        lambda: SimpleNamespace(is_job_running=lambda: True),
    )

    with pytest.raises(DatabaseSwitchBusyError, match="active background jobs"):
        DatabaseSettingsService().switch_database(_sqlite(Path.cwd() / "unused.db"))
