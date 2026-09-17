from __future__ import annotations

import json

import pytest

from server.configurations.management import ConfigurationManager
from server.domain.settings import ApplicationSettingsPatch
from server.services.errors import InternalServiceError
from server.services.settings import SettingsService


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


@pytest.fixture(autouse=True)
def embedded_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDED_DATABASE", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)


def test_public_projection_excludes_infrastructure_and_hidden_inference_fields(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    response = SettingsService(ConfigurationManager(config_path)).get_settings()

    payload = response.model_dump(mode="json", by_alias=True)
    assert payload["values"] == {
        "global": {"seed": 123},
        "features": {"allow_local_filesystem_access": False},
        "jobs": {"polling_interval": 2.5},
        "inference": {"model_timeout": 120},
    }
    assert payload["defaults"] == {
        "global": {"seed": 42},
        "features": {"allow_local_filesystem_access": True},
        "jobs": {"polling_interval": 1.0},
        "inference": {"model_timeout": 600},
    }


def test_update_and_reset_change_only_public_fields(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    service = SettingsService(ConfigurationManager(config_path))

    updated = service.update_settings(
        ApplicationSettingsPatch.model_validate(
            {"global": {"seed": 987}, "inference": {"model_timeout": 901}}
        )
    )
    assert updated.values.global_settings.seed == 987
    assert updated.values.inference.model_timeout == 901

    reset = service.reset_settings()
    assert reset.values.global_settings.seed == 42
    assert reset.values.inference.model_timeout == 600
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["inference"]["device"] == "cpu"
    assert persisted["inference"]["hf_local_only"] is False
    assert persisted["inference"]["max_loaded_models"] == 1


def test_patch_rejects_empty_unknown_and_null_values() -> None:
    with pytest.raises(ValueError):
        ApplicationSettingsPatch.model_validate({})
    with pytest.raises(ValueError):
        ApplicationSettingsPatch.model_validate({"inference": {"device": "cuda"}})
    with pytest.raises(ValueError):
        ApplicationSettingsPatch.model_validate({"global": {"seed": None}})


def test_persistence_failure_is_translated_to_safe_service_error(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")
    service = SettingsService(ConfigurationManager(config_path))

    def fail(_patch) -> None:
        raise RuntimeError("path and secret details must not escape")

    monkeypatch.setattr(service.manager, "update_application_settings", fail)
    with pytest.raises(InternalServiceError, match="Unable to persist application settings"):
        service.update_settings(
            ApplicationSettingsPatch.model_validate({"global": {"seed": 321}})
        )
