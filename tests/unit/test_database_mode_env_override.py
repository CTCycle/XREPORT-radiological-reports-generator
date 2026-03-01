from __future__ import annotations

from XREPORT.server.configurations.server import build_database_settings


def _payload() -> dict[str, object]:
    return {
        "embedded_database": True,
        "engine": "postgres",
        "host": "json-host",
        "port": 5432,
        "database_name": "json_db",
        "username": "json_user",
        "password": "json_password",
        "ssl": False,
        "ssl_ca": None,
        "connect_timeout": 20,
        "insert_batch_size": 800,
    }


def test_db_embedded_env_override_takes_precedence(monkeypatch) -> None:
    payload = _payload()
    monkeypatch.setenv("DB_EMBEDDED", "false")
    monkeypatch.setenv("DB_HOST", "env-host")
    monkeypatch.setenv("DB_NAME", "env_db")
    monkeypatch.setenv("DB_USER", "env_user")

    settings = build_database_settings(payload)

    assert settings.embedded_database is False
    assert settings.engine == "postgres"
    assert settings.host == "env-host"
    assert settings.port == 5432


def test_external_db_env_fields_override_json(monkeypatch) -> None:
    payload = _payload()
    payload["embedded_database"] = False
    payload["host"] = "json-host"
    payload["port"] = 1000
    payload["database_name"] = "json_db"
    payload["username"] = "json_user"
    payload["password"] = "json_password"

    monkeypatch.setenv("DB_EMBEDDED", "false")
    monkeypatch.setenv("DB_ENGINE", "postgresql+psycopg")
    monkeypatch.setenv("DB_HOST", "env-host")
    monkeypatch.setenv("DB_PORT", "6543")
    monkeypatch.setenv("DB_NAME", "env_db")
    monkeypatch.setenv("DB_USER", "env_user")
    monkeypatch.setenv("DB_PASSWORD", "env_password")
    monkeypatch.setenv("DB_SSL", "true")
    monkeypatch.setenv("DB_SSL_CA", "/tmp/ca.pem")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT", "45")
    monkeypatch.setenv("DB_INSERT_BATCH_SIZE", "77")

    settings = build_database_settings(payload)

    assert settings.embedded_database is False
    assert settings.engine == "postgresql+psycopg"
    assert settings.host == "env-host"
    assert settings.port == 6543
    assert settings.database_name == "env_db"
    assert settings.username == "env_user"
    assert settings.password == "env_password"
    assert settings.ssl is True
    assert settings.ssl_ca == "/tmp/ca.pem"
    assert settings.connect_timeout == 45
    assert settings.insert_batch_size == 77


def test_external_db_json_fields_are_ignored_when_env_absent(monkeypatch) -> None:
    payload = _payload()
    payload["embedded_database"] = False
    monkeypatch.setenv("DB_EMBEDDED", "false")

    settings = build_database_settings(payload)

    assert settings.embedded_database is False
    assert settings.engine == "postgres"
    assert settings.host is None
    assert settings.database_name is None
    assert settings.username is None
    assert settings.password is None
