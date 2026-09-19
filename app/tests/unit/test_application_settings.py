from __future__ import annotations

from pathlib import Path
import json

import pytest
import sqlalchemy

import server.repositories.database.engine as database_engine
import server.repositories.database.initializer as initializer
from server.configurations.settings import DatabaseSettings
from server.domain.settings import ApplicationSettingsPatch
from server.repositories.application_settings import ApplicationSettingsRepository
from server.repositories.database import Database
from server.repositories.schemas import ApplicationSettingsRecord
from server.services.errors import InternalServiceError
from server.services.settings import SettingsService


###############################################################################
def _sqlite_settings() -> DatabaseSettings:
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
    )


###############################################################################
def _database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    path = tmp_path / "database.db"
    monkeypatch.setattr(initializer, "DATABASE_FILE_PATH", path)
    monkeypatch.setattr(database_engine, "DATABASE_FILE_PATH", path)
    settings = _sqlite_settings()
    initializer.initialize_database(settings)
    database = Database(settings)
    with database.transaction() as session:
        session.query(ApplicationSettingsRecord).update(
            {
                "global_seed": 123,
                "allow_local_filesystem_access": False,
                "job_polling_interval": 2.5,
                "inference_hf_local_only": False,
                "inference_device": "cpu",
                "inference_model_timeout": 120,
            }
        )
    return database


###############################################################################
def test_legacy_json_is_imported_once_during_alembic_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    path = data_root / "settings" / "configurations.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "global": {"seed": 123},
                "features": {"allow_local_filesystem_access": False},
                "jobs": {"polling_interval": 2.5},
                "inference": {
                    "hf_local_only": False,
                    "device": "cuda",
                    "max_loaded_models": 1,
                    "model_timeout": 120,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XREPORT_DESKTOP", "true")
    monkeypatch.setenv("XREPORT_DATA_ROOT", str(data_root))
    db_path = tmp_path / "database.db"
    monkeypatch.setattr(initializer, "DATABASE_FILE_PATH", db_path)
    monkeypatch.setattr(database_engine, "DATABASE_FILE_PATH", db_path)
    settings = _sqlite_settings()
    initializer.initialize_database(settings)
    database = Database(settings)

    values = ApplicationSettingsRepository(database).get_settings()
    assert values.global_seed == 123
    assert values.allow_local_filesystem_access is False
    assert values.job_polling_interval == 2.5
    assert values.inference_hf_local_only is False
    assert values.inference_device == "cuda"
    assert values.inference_model_timeout == 120
    assert path.is_file()


###############################################################################
def test_legacy_json_with_non_singleton_model_limit_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    path = data_root / "settings" / "configurations.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "global": {"seed": 42},
                "features": {"allow_local_filesystem_access": True},
                "jobs": {"polling_interval": 1.0},
                "inference": {
                    "hf_local_only": True,
                    "device": "auto",
                    "max_loaded_models": 2,
                    "model_timeout": 600,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XREPORT_DESKTOP", "true")
    monkeypatch.setenv("XREPORT_DATA_ROOT", str(data_root))
    db_path = tmp_path / "database.db"
    monkeypatch.setattr(initializer, "DATABASE_FILE_PATH", db_path)
    monkeypatch.setattr(database_engine, "DATABASE_FILE_PATH", db_path)
    with pytest.raises(RuntimeError, match="max_loaded_models"):
        initializer.initialize_database(_sqlite_settings())
    assert path.is_file()
    if db_path.exists():
        engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
        try:
            assert "application_settings" not in sqlalchemy.inspect(engine).get_table_names()
        finally:
            engine.dispose()


###############################################################################
def test_fresh_upgrade_without_legacy_json_uses_typed_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("XREPORT_DESKTOP", "true")
    monkeypatch.setenv("XREPORT_DATA_ROOT", str(data_root))
    db_path = tmp_path / "database.db"
    monkeypatch.setattr(initializer, "DATABASE_FILE_PATH", db_path)
    monkeypatch.setattr(database_engine, "DATABASE_FILE_PATH", db_path)

    initializer.initialize_database(_sqlite_settings())
    values = ApplicationSettingsRepository(Database(_sqlite_settings())).get_settings()

    assert values.model_dump() == {
        "global_seed": 42,
        "allow_local_filesystem_access": True,
        "job_polling_interval": 1.0,
        "inference_hf_local_only": True,
        "inference_device": "auto",
        "inference_model_timeout": 600,
    }


###############################################################################
def test_public_projection_excludes_private_inference_policy(tmp_path, monkeypatch) -> None:
    database = _database(tmp_path, monkeypatch)
    response = SettingsService(ApplicationSettingsRepository(database)).get_settings()

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


###############################################################################
def test_update_and_reset_preserve_private_inference_policy(tmp_path, monkeypatch) -> None:
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
    assert persisted.inference_device == "cpu"
    assert persisted.inference_hf_local_only is False


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
