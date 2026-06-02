from __future__ import annotations

import pytest

from server.domain.settings import JsonServerSettings


@pytest.fixture(autouse=True)
def clear_database_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "XREPORT_DB_EMBEDDED",
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


def _build_database_settings():
    settings = JsonServerSettings.model_validate({})
    return settings.to_server_settings().database


def test_db_embedded_env_uses_sqlite_defaults() -> None:
    settings = _build_database_settings()

    assert settings.embedded_database is True
    assert settings.engine is None
    assert settings.host is None
    assert settings.port is None


def test_external_db_uses_component_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XREPORT_DB_EMBEDDED", "false")
    monkeypatch.setenv("XREPORT_DB_ENGINE", "postgres")
    monkeypatch.setenv("XREPORT_DB_HOST", "env-host")
    monkeypatch.setenv("XREPORT_DB_PORT", "1000")
    monkeypatch.setenv("XREPORT_DB_NAME", "env_db")
    monkeypatch.setenv("XREPORT_DB_USERNAME", "env_user")
    monkeypatch.setenv("XREPORT_DB_PASSWORD", "env_password")
    monkeypatch.setenv("XREPORT_DB_SSL", "true")
    monkeypatch.setenv("XREPORT_DB_SSL_CA", "/tmp/ca.pem")
    monkeypatch.setenv("XREPORT_DB_CONNECT_TIMEOUT", "45")
    monkeypatch.setenv("XREPORT_DB_INSERT_BATCH_SIZE", "77")

    settings = _build_database_settings()

    assert settings.embedded_database is False
    assert settings.engine == "postgres"
    assert settings.host == "env-host"
    assert settings.port == 1000
    assert settings.database_name == "env_db"
    assert settings.username == "env_user"
    assert settings.password == "env_password"
    assert settings.ssl is True
    assert settings.ssl_ca == "/tmp/ca.pem"
    assert settings.connect_timeout == 45
    assert settings.insert_batch_size == 77


def test_external_db_merges_database_url_with_component_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XREPORT_DB_EMBEDDED", "false")
    monkeypatch.setenv(
        "XREPORT_DATABASE_URL",
        "postgresql+psycopg://url_user:url_password@url-host:6789/url_db",
    )
    monkeypatch.setenv("XREPORT_DB_PORT", "1000")
    monkeypatch.setenv("XREPORT_DB_PASSWORD", "env_password")

    settings = _build_database_settings()

    assert settings.embedded_database is False
    assert settings.engine == "postgresql+psycopg"
    assert settings.host == "url-host"
    assert settings.port == 1000
    assert settings.database_name == "url_db"
    assert settings.username == "url_user"
    assert settings.password == "env_password"
