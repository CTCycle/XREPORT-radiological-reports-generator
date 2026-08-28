from __future__ import annotations

import json

import pytest

from server.configurations.management import ConfigurationManager

###############################################################################
def _configuration_payload() -> dict[str, object]:
    return {
        "global": {"seed": 123},
        "features": {"allow_local_filesystem_access": False},
        "jobs": {"polling_interval": 2.5},
        "database": {"backend": "postgresql", "host": "ignored-json-host"},
    }

###############################################################################
@pytest.fixture(autouse=True)
def clear_database_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "EMBEDDED_DATABASE",
        "DATABASE_URL",
        "DATABASE_ENGINE",
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_NAME",
        "DATABASE_USERNAME",
        "DATABASE_PASSWORD",
        "DATABASE_SSL",
        "DATABASE_SSL_CA",
        "DATABASE_CONNECT_TIMEOUT",
        "DATABASE_INSERT_BATCH_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)

###############################################################################
def test_configuration_uses_json_for_application_settings_and_env_for_database(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    monkeypatch.setenv("EMBEDDED_DATABASE", "false")
    monkeypatch.setenv("DATABASE_HOST", "env-host")
    monkeypatch.setenv("DATABASE_PORT", "15432")
    monkeypatch.setenv("DATABASE_NAME", "env-db")
    monkeypatch.setenv("DATABASE_USERNAME", "env-user")
    monkeypatch.setenv("DATABASE_PASSWORD", "env-password")

    settings = ConfigurationManager(config_path=str(config_path)).get_all()

    assert settings.global_settings.seed == 123
    assert settings.features.allow_local_filesystem_access is False
    assert settings.jobs.polling_interval == 2.5
    assert settings.database.backend == "postgresql"
    assert settings.database.host == "env-host"
    assert settings.database.port == 15432
    assert settings.database.database_name == "env-db"
    assert settings.database.username == "env-user"
    assert settings.database.password == "env-password"

###############################################################################
def test_configuration_rejects_invalid_files(tmp_path) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{invalid-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unable to load configuration"):
        ConfigurationManager(config_path=str(invalid_json))

    invalid_root = tmp_path / "invalid-root.json"
    invalid_root.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Configuration must be a JSON object"):
        ConfigurationManager(config_path=str(invalid_root))
