from __future__ import annotations

import json

import pytest

import server.configurations.management as management
from server.configurations.management import ConfigurationManager

###############################################################################
def _configuration_payload() -> dict[str, object]:
    return {
        "global": {"seed": 123},
        "features": {"allow_local_filesystem_access": False},
        "jobs": {"polling_interval": 2.5},
        "inference": {
            "hf_local_only": False,
            "device": "cpu",
            "max_loaded_models": 1,
            "model_timeout": 120,
        },
    }

###############################################################################
@pytest.fixture(autouse=True)
def clear_database_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "EMBEDDED_DATABASE",
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
        "DATABASE_URL",
        "HF_LOCAL_ONLY",
        "INFERENCE_DEVICE",
        "INFERENCE_MAX_LOADED_MODELS",
        "INFERENCE_MODEL_TIMEOUT",
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
    monkeypatch.setenv("DATABASE_ENGINE", "postgresql")
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
    assert settings.inference.hf_local_only is False
    assert settings.inference.device == "cpu"
    assert settings.inference.model_timeout == 120

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

###############################################################################
def test_inference_settings_are_owned_by_json_not_environment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    monkeypatch.setenv("HF_LOCAL_ONLY", "true")
    monkeypatch.setenv("INFERENCE_DEVICE", "cuda")
    monkeypatch.setenv("INFERENCE_MODEL_TIMEOUT", "999")

    settings = ConfigurationManager(config_path=str(config_path)).get_all()

    assert settings.inference.hf_local_only is False
    assert settings.inference.device == "cpu"
    assert settings.inference.model_timeout == 120

###############################################################################
def test_application_configuration_rejects_unknown_keys(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    payload = _configuration_payload()
    payload["unknown"] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid application configuration"):
        ConfigurationManager(config_path=str(config_path))

###############################################################################
def test_database_url_is_not_an_alternate_connection_source(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@db/XREPORT")

    with pytest.raises(ValueError, match="DATABASE_URL is not supported"):
        ConfigurationManager(config_path=str(config_path))

###############################################################################
def test_invalid_environment_values_fail_instead_of_using_defaults(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    monkeypatch.setenv("EMBEDDED_DATABASE", "maybe")

    with pytest.raises(ValueError, match="EMBEDDED_DATABASE must be a boolean"):
        ConfigurationManager(config_path=str(config_path))

###############################################################################
def test_partial_application_update_preserves_hidden_settings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDED_DATABASE", "true")
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    manager = ConfigurationManager(config_path=config_path)

    updated = manager.update_application_settings({"inference": {"model_timeout": 900}})

    assert updated.inference.model_timeout == 900
    assert updated.inference.device == "cpu"
    assert updated.inference.hf_local_only is False
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["inference"] == {
        "hf_local_only": False,
        "device": "cpu",
        "max_loaded_models": 1,
        "model_timeout": 900,
    }

###############################################################################
def test_sequential_partial_updates_do_not_overwrite_each_other(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDED_DATABASE", "true")
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    manager = ConfigurationManager(config_path=config_path)

    manager.update_application_settings({"global": {"seed": 321}})
    manager.update_application_settings({"jobs": {"polling_interval": 4.5}})

    current = manager.get_all()
    assert current.global_settings.seed == 321
    assert current.jobs.polling_interval == 4.5

###############################################################################
def test_failed_atomic_write_preserves_file_and_cached_settings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDED_DATABASE", "true")
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    manager = ConfigurationManager(config_path=config_path)
    original_text = config_path.read_text(encoding="utf-8")

    def fail_replace(_source, _target) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(management.os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="Unable to persist application configuration"):
        manager.update_application_settings({"global": {"seed": 999}})

    assert config_path.read_text(encoding="utf-8") == original_text
    assert manager.get_all().global_settings.seed == 123
