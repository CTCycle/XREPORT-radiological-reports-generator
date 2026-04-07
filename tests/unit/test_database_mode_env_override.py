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


def test_db_embedded_payload_uses_sqlite_defaults() -> None:
    payload = _payload()

    settings = build_database_settings(payload)

    assert settings.embedded_database is True
    assert settings.engine is None
    assert settings.host is None
    assert settings.port is None


def test_external_db_uses_payload_values() -> None:
    payload = _payload()
    payload["embedded_database"] = False
    payload["host"] = "json-host"
    payload["port"] = 1000
    payload["database_name"] = "json_db"
    payload["username"] = "json_user"
    payload["password"] = "json_password"
    payload["ssl"] = True
    payload["ssl_ca"] = "/tmp/ca.pem"
    payload["connect_timeout"] = 45
    payload["insert_batch_size"] = 77

    settings = build_database_settings(payload)

    assert settings.embedded_database is False
    assert settings.engine == "postgres"
    assert settings.host == "json-host"
    assert settings.port == 1000
    assert settings.database_name == "json_db"
    assert settings.username == "json_user"
    assert settings.password == "json_password"
    assert settings.ssl is True
    assert settings.ssl_ca == "/tmp/ca.pem"
    assert settings.connect_timeout == 45
    assert settings.insert_batch_size == 77


def test_external_db_normalizes_engine_value() -> None:
    payload = _payload()
    payload["embedded_database"] = False
    payload["engine"] = "PoStGreSQL+PsYcOpG"

    settings = build_database_settings(payload)

    assert settings.embedded_database is False
    assert settings.engine == "postgresql+psycopg"
