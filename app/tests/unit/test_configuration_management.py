from __future__ import annotations

import json

import pytest

from server.configurations.management import ConfigurationManager
from server.configurations.settings import JsonServerSettings

###############################################################################
def _configuration_payload() -> dict[str, object]:
    return {
        "global": {"seed": 123},
        "features": {"allow_local_filesystem_access": False},
        "jobs": {"polling_interval": 2.5},
    }

###############################################################################
@pytest.fixture(autouse=True)
def clear_database_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "XREPORT_DB_BACKEND",
        "XREPORT_DATABASE_URL",
        "XREPORT_DB_ENGINE",
        "XREPORT_DB_HOST",
        "XREPORT_DB_PORT",
        "XREPORT_DB_NAME",
        "XREPORT_DB_USERNAME",
        "XREPORT_DB_PASSWORD",
        "XREPORT_DB_SSL",
        "XREPORT_DB_SSL_CA",
        "XREPORT_DB_CONNECT_TIMEOUT",
        "XREPORT_DB_INSERT_BATCH_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)

###############################################################################
def test_configuration_manager_loads_and_accesses_values(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")

    manager = ConfigurationManager(config_path=str(config_path))

    all_settings = manager.get_all()
    assert all_settings.global_settings.seed == 123
    assert all_settings.features.allow_local_filesystem_access is False
    assert all_settings.jobs.polling_interval == 2.5
    assert all_settings.database.backend == "sqlite"

    assert manager.get_block("global") == {"seed": 123}
    assert manager.get_block("global_settings") == {"seed": 123}
    assert manager.get_value("global", "seed") == 123
    assert manager.get_value("missing", "key", default="fallback") == "fallback"

###############################################################################
def test_configuration_manager_resolves_database_from_env(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    monkeypatch.setenv("XREPORT_DB_BACKEND", "postgresql")
    monkeypatch.setenv("XREPORT_DB_HOST", "env-host")
    monkeypatch.setenv("XREPORT_DB_PORT", "15432")
    monkeypatch.setenv("XREPORT_DB_NAME", "env-db")
    monkeypatch.setenv("XREPORT_DB_USERNAME", "env-user")
    monkeypatch.setenv("XREPORT_DB_PASSWORD", "env-password")

    manager = ConfigurationManager(config_path=str(config_path))

    database_settings = manager.get_all().database
    assert database_settings.backend == "postgresql"
    assert database_settings.host == "env-host"
    assert database_settings.port == 15432
    assert database_settings.database_name == "env-db"
    assert database_settings.username == "env-user"
    assert database_settings.password == "env-password"

###############################################################################
def test_configuration_manager_ignores_database_block_in_json(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "configurations.json"
    payload = _configuration_payload() | {
        "database": {
            "backend": "postgresql",
            "host": "json-host",
            "port": 9999,
            "database_name": "json-db",
            "username": "json-user",
            "password": "json-password",
        }
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("XREPORT_DB_BACKEND", "sqlite")

    manager = ConfigurationManager(config_path=str(config_path))

    database_settings = manager.get_all().database
    assert database_settings.backend == "sqlite"
    assert database_settings.host is None
    assert database_settings.port is None
    assert manager.get_block("database")["backend"] == "sqlite"
    assert manager.get_block("database")["host"] is None

###############################################################################
def test_configuration_manager_reload_updates_snapshot(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    payload = _configuration_payload()
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    manager = ConfigurationManager(config_path=str(config_path))

    payload["global"] = {"seed": 999}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = manager.reload()
    assert reloaded.global_settings.seed == 999
    assert manager.get_all().global_settings.seed == 999

###############################################################################
def test_configuration_manager_raises_on_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="Configuration file not found"):
        ConfigurationManager(config_path=str(missing_path))

###############################################################################
def test_configuration_manager_raises_on_invalid_json(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text("{invalid-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unable to load configuration"):
        ConfigurationManager(config_path=str(config_path))

###############################################################################
def test_configuration_manager_raises_when_json_root_is_not_object(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Configuration must be a JSON object"):
        ConfigurationManager(config_path=str(config_path))


def test_database_defaults_are_sqlite() -> None:
    settings = JsonServerSettings.model_validate({}).to_server_settings().database

    assert settings.backend == "sqlite"
    assert settings.engine is None
    assert settings.host is None
    assert settings.port is None


def test_database_url_merge_with_component_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XREPORT_DB_BACKEND", "postgresql")
    monkeypatch.setenv(
        "XREPORT_DATABASE_URL",
        "postgresql+psycopg://url_user:url_password@url-host:6789/url_db",
    )
    monkeypatch.setenv("XREPORT_DB_PORT", "1000")
    monkeypatch.setenv("XREPORT_DB_PASSWORD", "env_password")

    settings = JsonServerSettings.model_validate({}).to_server_settings().database

    assert settings.backend == "postgresql"
    assert settings.engine == "postgresql+psycopg"
    assert settings.host == "url-host"
    assert settings.port == 1000
    assert settings.database_name == "url_db"
    assert settings.username == "url_user"
    assert settings.password == "env_password"
