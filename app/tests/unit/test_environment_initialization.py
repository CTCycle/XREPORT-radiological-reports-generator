from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import server.configurations.database_config as database_config
import server.configurations.environment as environment
from server.configurations.runtime_environment import get_runtime_environment_settings

###############################################################################
@pytest.fixture(autouse=True)
def reset_environment_state() -> None:
    environment._environment_state.cache_clear()
    database_config.clear_database_configuration_error()
    yield
    environment._environment_state.cache_clear()
    database_config.clear_database_configuration_error()

###############################################################################
def _configure_environment_paths(tmp_path, monkeypatch: pytest.MonkeyPatch):
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    env_path = settings_dir / ".env"
    example_path = settings_dir / ".env.example"
    monkeypatch.setattr(environment, "ENV_FILE_PATH", env_path)
    monkeypatch.setattr(environment, "ENV_EXAMPLE_FILE_PATH", example_path)
    monkeypatch.setattr(database_config, "DATABASE_CONFIG_FILE_PATH", settings_dir.parent / "database-config.json")
    monkeypatch.setattr(database_config, "DATABASE_FILE_PATH", settings_dir.parent / "database.db")
    return env_path, example_path

###############################################################################
def test_missing_env_is_seeded_from_template_and_loaded(tmp_path, monkeypatch) -> None:
    env_path, example_path = _configure_environment_paths(tmp_path, monkeypatch)
    contents = b"XREPORT_TEST_ENV=from-template\n"
    example_path.write_bytes(contents)
    monkeypatch.delenv("XREPORT_TEST_ENV", raising=False)

    loaded_path = environment.load_environment(force=True)

    assert loaded_path == env_path
    assert env_path.read_bytes() == contents
    assert os.environ["XREPORT_TEST_ENV"] == "from-template"

###############################################################################
def test_existing_env_is_never_overwritten_by_template(tmp_path, monkeypatch) -> None:
    env_path, example_path = _configure_environment_paths(tmp_path, monkeypatch)
    env_path.write_bytes(b"XREPORT_TEST_ENV=existing\n")
    example_path.write_bytes(b"XREPORT_TEST_ENV=template\n")
    monkeypatch.delenv("XREPORT_TEST_ENV", raising=False)

    environment.load_environment(force=True)

    assert env_path.read_bytes() == b"XREPORT_TEST_ENV=existing\n"
    assert os.environ["XREPORT_TEST_ENV"] == "existing"

###############################################################################
def test_explicit_resource_root_override_survives_environment_loading(
    tmp_path, monkeypatch
) -> None:
    env_path, _ = _configure_environment_paths(tmp_path, monkeypatch)
    env_path.write_text(
        "XREPORT_RESOURCES_DIR=from-file\nXREPORT_TEST_ENV=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XREPORT_RESOURCES_DIR", "from-process")
    monkeypatch.setenv("XREPORT_TEST_ENV", "from-process")

    environment.load_environment(force=True)

    assert os.environ["XREPORT_RESOURCES_DIR"] == "from-process"
    assert os.environ["XREPORT_TEST_ENV"] == "from-file"

###############################################################################
def test_stray_database_keys_are_ignored_and_purged(tmp_path, monkeypatch) -> None:
    env_path, _ = _configure_environment_paths(tmp_path, monkeypatch)
    env_path.write_text(
        "EMBEDDED_DATABASE=false\nDATABASE_INSERT_BATCH_SIZE=25\nXREPORT_TEST_ENV=kept\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_HOST", "leftover-host")
    monkeypatch.setenv("EMBEDDED_DATABASE", "false")
    monkeypatch.delenv("XREPORT_TEST_ENV", raising=False)

    environment.load_environment(force=True)

    configured = database_config.read_database_configuration()
    assert configured.backend == "sqlite"
    assert configured.sqlite_path == env_path.parent.parent / "database.db"
    assert configured.insert_batch_size == 1000
    assert "DATABASE_HOST" not in os.environ
    assert "EMBEDDED_DATABASE" not in os.environ
    assert os.environ["XREPORT_TEST_ENV"] == "kept"
    assert "DATABASE_INSERT_BATCH_SIZE" in env_path.read_text(encoding="utf-8")


###############################################################################
def test_supported_runtime_settings_are_loaded_from_dotenv(tmp_path, monkeypatch) -> None:
    env_path, _ = _configure_environment_paths(tmp_path, monkeypatch)
    env_path.write_text(
        "\n".join(
            (
                "INFERENCE_HF_LOCAL_ONLY=false",
                "INFERENCE_DEVICE=cpu",
                "CORS_ALLOWED_ORIGINS=https://ui.example,https://desktop.example",
                "CORS_ALLOW_CREDENTIALS=true",
                "CORS_ALLOW_METHODS=GET,OPTIONS",
                "CORS_ALLOW_HEADERS=Authorization,X-XREPORT-Desktop-Token",
                "",
            )
        ),
        encoding="utf-8",
    )

    environment.load_environment(force=True)
    settings = get_runtime_environment_settings()

    assert settings.inference_hf_local_only is False
    assert settings.inference_device == "cpu"
    assert settings.cors.allowed_origins == (
        "https://ui.example",
        "https://desktop.example",
    )
    assert settings.cors.allow_credentials is True
    assert settings.cors.allow_methods == ("GET", "OPTIONS")
    assert settings.cors.allow_headers == (
        "Authorization",
        "X-XREPORT-Desktop-Token",
    )

    from server.app import create_app

    response = TestClient(create_app()).options(
        "/api/health",
        headers={
            "Origin": "https://ui.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-XREPORT-Desktop-Token",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://ui.example"
