from __future__ import annotations

from pathlib import Path

import pytest

import server.repositories.database.initializer as initializer
from server.configurations.settings import DatabaseSettings
from server.domain.settings import ApplicationSettingsPatch
from server.repositories.application_settings import ApplicationSettingsRepository
from server.repositories.database import Database
from server.repositories.schemas import ApplicationSettingsRecord
from server.services.errors import InternalServiceError
from server.services.settings import SettingsService

###############################################################################
def _sqlite_settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(
        backend="sqlite",
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=3,
        insert_batch_size=1000,
        sqlite_path=path,
    )

###############################################################################
def _database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    path = tmp_path / "database.db"
    settings = _sqlite_settings(path)
    initializer.initialize_database(settings)
    database = Database(settings)
    with database.transaction() as session:
        session.query(ApplicationSettingsRecord).update(
            {
                "global_seed": 123,
                "allow_local_filesystem_access": False,
                "job_polling_interval": 2.5,
                "inference_model_timeout": 120,
            }
        )
    return database

###############################################################################
def test_fresh_upgrade_uses_typed_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("XREPORT_DESKTOP", "true")
    monkeypatch.setenv("XREPORT_DATA_ROOT", str(data_root))
    db_path = tmp_path / "database.db"
    initializer.initialize_database(_sqlite_settings(db_path))
    values = ApplicationSettingsRepository(Database(_sqlite_settings(db_path))).get_settings()

    assert values.model_dump() == {
        "global_seed": 42,
        "allow_local_filesystem_access": True,
        "job_polling_interval": 1.0,
        "inference_model_timeout": 600,
        "theme_preference": None,
    }

###############################################################################
def test_public_projection_contains_only_application_settings(tmp_path, monkeypatch) -> None:
    database = _database(tmp_path, monkeypatch)
    response = SettingsService(ApplicationSettingsRepository(database)).get_settings()

    payload = response.model_dump(mode="json", by_alias=True)
    assert payload["values"] == {
        "global": {"seed": 123},
        "features": {"allow_local_filesystem_access": False},
        "jobs": {"polling_interval": 2.5},
        "inference": {"model_timeout": 120},
        "theme_preference": None,
    }
    assert payload["defaults"] == {
        "global": {"seed": 42},
        "features": {"allow_local_filesystem_access": True},
        "jobs": {"polling_interval": 1.0},
        "inference": {"model_timeout": 600},
        "theme_preference": None,
    }

###############################################################################
def test_update_and_reset_change_application_settings(tmp_path, monkeypatch) -> None:
    database = _database(tmp_path, monkeypatch)
    repository = ApplicationSettingsRepository(database)
    service = SettingsService(repository)

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
    persisted = repository.get_settings()
    assert persisted.theme_preference is None

###############################################################################
def test_theme_preference_is_saved_and_reset_with_active_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _database(tmp_path, monkeypatch)
    repository = ApplicationSettingsRepository(database)
    service = SettingsService(repository)

    updated = service.update_settings(
        ApplicationSettingsPatch.model_validate({"theme_preference": "dark"})
    )
    assert updated.values.theme_preference == "dark"
    assert repository.get_settings().theme_preference == "dark"

    reset = service.reset_settings()
    assert reset.values.theme_preference is None
    assert repository.get_settings().theme_preference is None

###############################################################################
def test_public_settings_are_restored_after_reopening_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _database(tmp_path, monkeypatch)
    path = tmp_path / "database.db"
    repository = ApplicationSettingsRepository(database)
    repository.update_public_settings(
        {
            "global_seed": 321,
            "allow_local_filesystem_access": True,
            "job_polling_interval": 3.5,
            "inference_model_timeout": 777,
            "theme_preference": "dark",
        }
    )
    database.engine.dispose()

    reopened = Database(_sqlite_settings(path))
    try:
        restored = ApplicationSettingsRepository(reopened).get_settings()
        assert restored.global_seed == 321
        assert restored.allow_local_filesystem_access is True
        assert restored.job_polling_interval == 3.5
        assert restored.inference_model_timeout == 777
        assert restored.theme_preference == "dark"
    finally:
        reopened.engine.dispose()

###############################################################################
def test_patch_rejects_empty_unknown_and_null_values() -> None:
    with pytest.raises(ValueError):
        ApplicationSettingsPatch.model_validate({})
    with pytest.raises(ValueError):
        ApplicationSettingsPatch.model_validate({"inference": {"device": "cuda"}})
    with pytest.raises(ValueError):
        ApplicationSettingsPatch.model_validate({"global": {"seed": None}})

###############################################################################
def test_repository_rejects_private_columns(tmp_path, monkeypatch) -> None:
    repository = ApplicationSettingsRepository(_database(tmp_path, monkeypatch))
    with pytest.raises(ValueError, match="Unsupported application setting"):
        repository.update_public_settings({"inference_device": "cuda"})

###############################################################################
def test_repository_validates_candidate_before_persisting(tmp_path, monkeypatch) -> None:
    repository = ApplicationSettingsRepository(_database(tmp_path, monkeypatch))
    with pytest.raises(ValueError):
        repository.update_public_settings({"global_seed": -1})
    assert repository.get_settings().global_seed == 123

###############################################################################
def test_persistence_failure_is_translated_to_safe_service_error(
    tmp_path, monkeypatch
) -> None:
    database = _database(tmp_path, monkeypatch)
    service = SettingsService(ApplicationSettingsRepository(database))

    def fail(_changes) -> None:
        raise RuntimeError("path and secret details must not escape")

    monkeypatch.setattr(service.repository, "update_public_settings", fail)
    with pytest.raises(InternalServiceError, match="Unable to persist application settings"):
        service.update_settings(
            ApplicationSettingsPatch.model_validate({"global": {"seed": 321}})
        )
