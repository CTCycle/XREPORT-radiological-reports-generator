from __future__ import annotations

import json

import pytest

from XREPORT.server.configurations.management import ConfigurationManager


def _configuration_payload() -> dict[str, object]:
    return {
        "global": {"seed": 123},
        "features": {"allow_local_filesystem_access": False},
        "database": {
            "embedded_database": True,
            "engine": "postgres",
            "host": "127.0.0.1",
            "port": 5432,
            "database_name": "XREPORT",
            "username": "postgres",
            "password": "",
            "ssl": False,
            "ssl_ca": None,
            "connect_timeout": 30,
            "insert_batch_size": 1000,
        },
        "jobs": {"polling_interval": 2.5},
    }


def test_configuration_manager_loads_and_accesses_values(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps(_configuration_payload()), encoding="utf-8")

    manager = ConfigurationManager(config_path=str(config_path))

    all_settings = manager.get_all()
    assert all_settings.global_settings.seed == 123
    assert all_settings.features.allow_local_filesystem_access is False
    assert all_settings.jobs.polling_interval == 2.5

    assert manager.get_block("global") == {"seed": 123}
    assert manager.get_block("global_settings") == {"seed": 123}
    assert manager.get_value("global", "seed") == 123
    assert manager.get_value("missing", "key", default="fallback") == "fallback"


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


def test_configuration_manager_raises_on_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="Configuration file not found"):
        ConfigurationManager(config_path=str(missing_path))


def test_configuration_manager_raises_on_invalid_json(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text("{invalid-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unable to load configuration"):
        ConfigurationManager(config_path=str(config_path))


def test_configuration_manager_raises_when_json_root_is_not_object(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Configuration must be a JSON object"):
        ConfigurationManager(config_path=str(config_path))
