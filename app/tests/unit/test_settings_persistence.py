from __future__ import annotations

from pathlib import Path
import pytest
import xreport_shared.repositories.database.initializer as initializer
import json
import os
import xreport_shared.configurations.database_config as database_config
import xreport_shared.repositories.database.backend as database_backend
import xreport_shared.services.database_settings as database_settings_service
import sqlalchemy
import xreport_shared.repositories.database.engine as database_engine
import asyncio
from xreport_shared.configurations.settings import DatabaseSettings
from xreport_shared.domain.settings import ApplicationSettingsPatch
from xreport_shared.repositories.application_settings import ApplicationSettingsRepository
from xreport_shared.repositories.database import Database as ApplicationDatabase
from xreport_shared.repositories.database.schema import ALEMBIC_INI, ApplicationSchema, application_schema
from xreport_inference.repositories.schemas.inference import InferenceSettingsRecord
from xreport_shared.services.errors import BadRequestError, InternalServiceError
from xreport_shared.services.settings import SettingsService
from types import SimpleNamespace
from pydantic import ValidationError
from xreport_shared.configurations.database_config import DatabaseConfiguration, DatabaseConfigurationError
from xreport_shared.repositories.database.engine import Database
from xreport_shared.services.database_settings import DatabaseSwitchBusyError, DatabaseSettingsService
from concurrent.futures import ThreadPoolExecutor
from alembic import command
from alembic.config import Config
from xreport_inference.repositories.schemas.inference import InferenceReport, InferenceRun
from xreport_shared.common.database_switch import DatabaseRequestGate














###############################################################################
def _application_sqlite_settings(path: Path) -> DatabaseSettings:
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
def _database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ApplicationDatabase:
    path = tmp_path / "database.db"
    settings = _application_sqlite_settings(path)
    initializer.initialize_database(settings)
    database = ApplicationDatabase(settings)
    with database.transaction() as session:
        session.query(InferenceSettingsRecord).update(
            {
                "global_seed": 123,
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
    initializer.initialize_database(_application_sqlite_settings(db_path))
    values = ApplicationSettingsRepository(ApplicationDatabase(_application_sqlite_settings(db_path))).get_settings()

    assert values.model_dump() == {
        "global_seed": 42,
        "job_polling_interval": 1.0,
        "inference_model_timeout": 600,
        "inference_model_idle_timeout": 300,
        **{f"inference_{name}": 0 for name in _GENERATION_DEFAULTS},
        "theme_preference": None,
    }

_GENERATION_DEFAULTS = {
    "max_new_tokens_deterministic": 0,
    "max_new_tokens_concise": 0,
    "max_new_tokens_detailed": 0,
    "detailed_num_beams": 0,
}

###############################################################################
def test_public_projection_contains_only_application_settings(tmp_path, monkeypatch) -> None:
    database = _database(tmp_path, monkeypatch)
    response = SettingsService(ApplicationSettingsRepository(database)).get_settings()

    payload = response.model_dump(mode="json", by_alias=True)
    assert payload["values"] == {
        "global": {"seed": 123},
        "jobs": {"polling_interval": 2.5},
        "inference": {"model_timeout": 120, "model_idle_timeout": 300, **_GENERATION_DEFAULTS},
        "theme_preference": None,
    }
    assert payload["defaults"] == {
        "global": {"seed": 42},
        "jobs": {"polling_interval": 1.0},
        "inference": {"model_timeout": 600, "model_idle_timeout": 300, **_GENERATION_DEFAULTS},
        "theme_preference": None,
    }

###############################################################################
@pytest.mark.usefixtures("studio_profile")
def test_studio_settings_have_no_inference_controls(tmp_path) -> None:
    settings = _application_sqlite_settings(tmp_path / "database.db")
    initializer.initialize_database(settings)
    service = SettingsService(ApplicationSettingsRepository(ApplicationDatabase(settings)))

    payload = service.update_settings(
        ApplicationSettingsPatch.model_validate({"global": {"seed": 7}})
    ).model_dump(mode="json", by_alias=True)
    assert payload["values"] == {
        "global": {"seed": 7},
        "jobs": {"polling_interval": 1.0},
        "inference": None,
        "theme_preference": None,
    }
    with pytest.raises(BadRequestError, match="no inference settings"):
        service.update_settings(
            ApplicationSettingsPatch.model_validate({"inference": {"model_timeout": 5}})
        )

###############################################################################
def test_generation_limits_persist_and_reject_out_of_range(tmp_path, monkeypatch) -> None:
    service = SettingsService(ApplicationSettingsRepository(_database(tmp_path, monkeypatch)))

    updated = service.update_settings(
        ApplicationSettingsPatch.model_validate(
            {"inference": {"max_new_tokens_concise": 200, "detailed_num_beams": 3}}
        )
    )
    assert updated.values.inference.max_new_tokens_concise == 200
    assert updated.values.inference.detailed_num_beams == 3
    for patch in ({"max_new_tokens_detailed": 4097}, {"detailed_num_beams": 9}):
        with pytest.raises(ValueError):
            ApplicationSettingsPatch.model_validate({"inference": patch})

###############################################################################
def test_update_and_reset_change_application_settings(tmp_path, monkeypatch) -> None:
    database = _database(tmp_path, monkeypatch)
    repository = ApplicationSettingsRepository(database)
    service = SettingsService(repository)

    updated = service.update_settings(
        ApplicationSettingsPatch.model_validate(
            {"global": {"seed": 987}, "inference": {"model_timeout": 901, "model_idle_timeout": 0}}
        )
    )
    assert updated.values.global_settings.seed == 987
    assert updated.values.inference.model_timeout == 901
    assert updated.values.inference.model_idle_timeout == 0

    reset = service.reset_settings()
    assert reset.values.global_settings.seed == 42
    assert reset.values.inference.model_timeout == 600
    assert reset.values.inference.model_idle_timeout == 300
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
            "job_polling_interval": 3.5,
            "inference_model_timeout": 777,
            "theme_preference": "dark",
        }
    )
    database.engine.dispose()

    reopened = ApplicationDatabase(_application_sqlite_settings(path))
    try:
        restored = ApplicationSettingsRepository(reopened).get_settings()
        assert restored.global_seed == 321
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






















###############################################################################
def _sqlite(path: Path) -> DatabaseConfiguration:
    return DatabaseConfiguration(backend="sqlite", sqlite_path=path)

###############################################################################
def _seed_database(settings, seed: int) -> None:
    initializer.initialize_database(settings)
    database = Database(settings)
    try:
        with database.transaction() as session:
            session.query(InferenceSettingsRecord).update({"global_seed": seed})
    finally:
        database.engine.dispose()

###############################################################################
def test_sqlite_configuration_requires_an_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="absolute"):
        DatabaseConfiguration(backend="sqlite", sqlite_path=Path("relative.db"))

###############################################################################
def test_postgres_configuration_requires_connection_identity() -> None:
    with pytest.raises(ValidationError, match="host"):
        DatabaseConfiguration(backend="postgresql")

###############################################################################
@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI is the credential store")
def test_database_config_protects_password_and_roundtrips(tmp_path: Path) -> None:
    configuration = DatabaseConfiguration(
        backend="postgresql",
        host="db.example.test",
        database_name="xreport",
        username="xreport",
        password="top-secret-value",
    )
    path = tmp_path / "database-config.json"

    database_config.save_database_configuration(configuration, path=path)
    saved = path.read_text(encoding="utf-8")
    payload = json.loads(saved)

    assert "top-secret-value" not in saved
    assert payload["password_protected"]
    assert "password" not in payload
    assert database_config.read_database_configuration(path=path).password == "top-secret-value"

###############################################################################
def test_invalid_database_config_is_preserved_for_recovery(tmp_path: Path) -> None:
    path = tmp_path / "database-config.json"
    path.write_text('{"schema_version":99,"backend":"sqlite"}\n', encoding="utf-8")
    original = path.read_bytes()

    status = database_config.read_database_configuration_status(path=path)

    assert status.configuration is None
    assert status.error
    assert path.read_bytes() == original

###############################################################################
def test_connection_test_reports_an_unavailable_postgres_server() -> None:
    configuration = DatabaseConfiguration(
        backend="postgresql",
        host="127.0.0.1",
        port=1,
        database_name="unavailable",
        username="xreport",
        connect_timeout=1,
    )

    result = DatabaseSettingsService().test_connection(configuration)

    assert result.success is False
    assert result.connected is False
    assert result.schema_ready is False
    assert result.detail

###############################################################################
def test_sqlite_switch_reloads_active_database_settings_and_can_roll_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_settings = _sqlite(tmp_path / "original.db").to_database_settings()
    target_configuration = _sqlite(tmp_path / "target.db")
    target_settings = target_configuration.to_database_settings()
    _seed_database(original_settings, 11)
    _seed_database(target_settings, 22)

    monkeypatch.setattr(
        database_settings_service,
        "get_job_manager",
        lambda: SimpleNamespace(is_job_running=lambda: False),
    )
    monkeypatch.setattr(
        "xreport_shared.services.startup_validation.run_startup_validations",
        lambda settings: settings,
    )
    config_path = tmp_path / "database-config.json"
    monkeypatch.setattr(
        database_settings_service,
        "save_database_configuration",
        lambda configuration: database_config.save_database_configuration(
            configuration, path=config_path
        ),
    )

    database_backend.reset_database()
    try:
        database_backend.activate_database(Database(original_settings))
        facade = database_backend.get_database()
        service = DatabaseSettingsService()

        result = service.switch_database(target_configuration)

        assert database_backend.get_database() is facade
        assert result.response.settings.values.global_settings.seed == 22
        assert ApplicationSettingsRepository().get_settings().global_seed == 22
        assert database_config.read_database_configuration(path=config_path) == target_configuration

        def fail_save(_configuration, **_kwargs) -> None:
            raise DatabaseConfigurationError("simulated configuration write failure")

        monkeypatch.setattr(database_settings_service, "save_database_configuration", fail_save)
        with pytest.raises(DatabaseConfigurationError, match="simulated configuration write failure"):
            service.switch_database(_sqlite(tmp_path / "original.db"))

        assert database_backend.get_database() is facade
        assert ApplicationSettingsRepository().get_settings().global_seed == 22
        assert database_config.read_database_configuration(path=config_path) == target_configuration
    finally:
        database_backend.reset_database()

###############################################################################
def test_database_switch_is_blocked_while_a_job_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        database_settings_service,
        "get_job_manager",
        lambda: SimpleNamespace(is_job_running=lambda: True),
    )

    with pytest.raises(DatabaseSwitchBusyError, match="active background jobs"):
        DatabaseSettingsService().switch_database(_sqlite(Path.cwd() / "unused.db"))

















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
def _postgres_settings() -> DatabaseSettings:
    return DatabaseSettings(
        backend="postgresql",
        engine="postgresql+psycopg",
        host="127.0.0.1",
        port=5432,
        database_name="xreport-test",
        username="xreport",
        password="password",
        ssl=False,
        ssl_ca=None,
        connect_timeout=1,
        insert_batch_size=1000,
    )

###############################################################################
APPLICATION_PROFILES = pytest.mark.parametrize("profile", ["xreport", "studio"])

###############################################################################
def _alembic_config(path: Path, schema: ApplicationSchema) -> Config:
    config = Config(
        str(ALEMBIC_INI),
        ini_section=schema.migrations_dir.name,
    )
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path.as_posix()}")
    return config

###############################################################################
def _database_tables(path: Path) -> set[str]:
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    try:
        return set(sqlalchemy.inspect(engine).get_table_names())
    finally:
        engine.dispose()

###############################################################################
def _schema_revision(path: Path, schema: ApplicationSchema) -> str:
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    try:
        with engine.connect() as connection:
            return connection.exec_driver_sql(
                f"SELECT version_num FROM {schema.version_table}"
            ).scalar_one()
    finally:
        engine.dispose()

###############################################################################
def _insert_checkpoint(path: Path, name: str) -> None:
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO checkpoints (name, name_key, path, created_at, last_seen_at) "
                "VALUES (?, ?, 'artifact', '2026-10-20 12:00:00', '2026-10-20 12:00:00')",
                (name, name),
            )
    finally:
        engine.dispose()

###############################################################################
@APPLICATION_PROFILES
def test_sqlite_startup_creates_only_the_application_schema(
    tmp_path, profile: str
) -> None:
    schema = application_schema(profile)
    database_path = tmp_path / "database.db"
    initializer.initialize_database(_sqlite_settings(database_path), schema)
    initializer.prepare_database_for_startup(_sqlite_settings(database_path), schema)

    assert _database_tables(database_path) == set(schema.metadata.tables) | {
        schema.version_table
    }
    assert _schema_revision(database_path, schema) == initializer.schema_head(schema)

###############################################################################
def test_application_schemas_do_not_share_application_tables() -> None:
    inference = set(application_schema("xreport").metadata.tables)
    studio = set(application_schema("studio").metadata.tables)
    # Each application keeps its own settings row and checkpoint registry.
    assert inference & studio == {"application_settings", "checkpoints"}
    assert {"inference_runs", "inference_reports"} <= inference - studio
    assert {"datasets", "checkpoint_evaluations"} <= studio - inference

###############################################################################
@APPLICATION_PROFILES
def test_database_at_head_preserves_settings_and_records(tmp_path, profile) -> None:
    schema = application_schema(profile)
    database_path = tmp_path / "database.db"
    command.upgrade(_alembic_config(database_path, schema), "heads")
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE application_settings SET global_seed = 909 WHERE settings_id = 1"
            )
    finally:
        engine.dispose()
    _insert_checkpoint(database_path, "retained")

    initializer.initialize_database(_sqlite_settings(database_path), schema)

    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql(
                "SELECT global_seed, theme_preference FROM application_settings "
                "WHERE settings_id = 1"
            ).one() == (909, None)
            assert connection.exec_driver_sql(
                "SELECT name FROM checkpoints"
            ).scalar_one() == "retained"
    finally:
        engine.dispose()
    assert _schema_revision(database_path, schema) == initializer.schema_head(schema)

###############################################################################
@pytest.mark.parametrize(
    ("profile", "owner"), [("xreport", "studio"), ("studio", "xreport")]
)
def test_database_of_the_other_application_is_rejected_without_modification(
    tmp_path, profile: str, owner: str
) -> None:
    database_path = tmp_path / "database.db"
    owner_schema = application_schema(owner)
    initializer.initialize_database(_sqlite_settings(database_path), owner_schema)
    tables = _database_tables(database_path)

    with pytest.raises(RuntimeError, match=f"belongs to {owner_schema.title}"):
        initializer.prepare_database_for_startup(
            _sqlite_settings(database_path), application_schema(profile)
        )

    assert _database_tables(database_path) == tables

###############################################################################
def test_retired_shared_schema_database_is_rejected(tmp_path) -> None:
    database_path = tmp_path / "database.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
            )
            connection.exec_driver_sql(
                "INSERT INTO alembic_version VALUES ('d4a9c6e2b3f1')"
            )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="retired shared XREPORT schema"):
        initializer.prepare_database_for_startup(_sqlite_settings(database_path))

    assert _database_tables(database_path) == {"alembic_version"}

###############################################################################
@APPLICATION_PROFILES
def test_unknown_database_revision_is_rejected_without_modification(
    tmp_path, profile: str
) -> None:
    schema = application_schema(profile)
    database_path = tmp_path / "database.db"
    command.upgrade(_alembic_config(database_path, schema), "heads")
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                f"UPDATE {schema.version_table} SET version_num = 'f48a7c2e91b6'"
            )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="Recreate or replace the database"):
        initializer.initialize_database(_sqlite_settings(database_path), schema)

    assert _schema_revision(database_path, schema) == "f48a7c2e91b6"

###############################################################################
@APPLICATION_PROFILES
def test_nonempty_unversioned_schema_is_rejected_without_implicit_adoption(
    tmp_path, profile: str
) -> None:
    schema = application_schema(profile)
    database_path = tmp_path / "database.db"
    command.upgrade(_alembic_config(database_path, schema), "heads")
    _insert_checkpoint(database_path, "unversioned")
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f"DROP TABLE {schema.version_table}")
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="implicit schema adoption"):
        initializer.initialize_database(_sqlite_settings(database_path), schema)

    assert schema.version_table not in _database_tables(database_path)

###############################################################################
def test_unknown_partial_schema_is_rejected_without_stamping(tmp_path) -> None:
    database_path = tmp_path / "database.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE checkpoints (checkpoint_id INTEGER PRIMARY KEY, name TEXT)"
            )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="implicit schema adoption"):
        initializer.prepare_database_for_startup(_sqlite_settings(database_path))

    assert _database_tables(database_path) == {"checkpoints"}

###############################################################################
def test_failed_migration_rolls_back_database_changes(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"

    def fail_upgrade(config: Config, _revision: str) -> None:
        connection = config.attributes["connection"]
        connection.exec_driver_sql("CREATE TABLE transient_failure (id INTEGER)")
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(initializer.command, "upgrade", fail_upgrade)
    with pytest.raises(RuntimeError, match="Unexpected error"):
        initializer.initialize_database(_sqlite_settings(database_path))

    assert _database_tables(database_path) == set()

###############################################################################
@APPLICATION_PROFILES
def test_head_schema_drift_blocks_startup(tmp_path, profile: str) -> None:
    schema = application_schema(profile)
    database_path = tmp_path / "database.db"
    initializer.initialize_database(_sqlite_settings(database_path), schema)
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "ALTER TABLE checkpoints ADD COLUMN unexpected TEXT"
            )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="differs from the current ORM schema"):
        initializer.prepare_database_for_startup(
            _sqlite_settings(database_path), schema
        )

###############################################################################
def test_sqlite_enforces_foreign_keys_and_cascades_report_deletion(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "database.db"
    initializer.initialize_database(_sqlite_settings(database_path))
    database = database_engine.Database(_sqlite_settings(database_path))

    try:
        with database.engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1

        with database.transaction() as session:
            inference_run = InferenceRun(
                provider="huggingface",
                model_ref="huggingface:validation/foreign-key-test",
                generation_profile="deterministic",
                generation_config_json={},
                request_id="foreign-key-parent",
                status="succeeded",
            )
            inference_run.reports.append(
                InferenceReport(
                    input_image_name="foreign-key.png",
                    input_image_name_key="foreign-key.png",
                    image_index=0,
                    generated_report="Findings\nFixture report.",
                )
            )
            session.add(inference_run)
            session.flush()
            inference_run_id = inference_run.inference_run_id

        with pytest.raises(sqlalchemy.exc.IntegrityError):
            with database.engine.begin() as connection:
                connection.exec_driver_sql(
                    "INSERT INTO inference_reports "
                    "(inference_run_id, input_image_name, input_image_name_key, "
                    "image_index, generated_report) VALUES (?, ?, ?, ?, ?)",
                    (
                        inference_run_id + 1,
                        "orphan.png",
                        "orphan.png",
                        1,
                        "Findings\nOrphan fixture.",
                    ),
                )

        with database.engine.begin() as connection:
            connection.exec_driver_sql(
                "DELETE FROM inference_runs WHERE inference_run_id = ?",
                (inference_run_id,),
            )
            remaining_reports = connection.exec_driver_sql(
                "SELECT count(*) FROM inference_reports WHERE inference_run_id = ?",
                (inference_run_id,),
            ).scalar_one()
            assert remaining_reports == 0
    finally:
        database.engine.dispose()

###############################################################################
def test_concurrent_sqlite_initialization_is_safe(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    settings = _sqlite_settings(database_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(initializer.initialize_database, settings) for _ in range(2)
        ]
        for future in futures:
            future.result()

    schema = application_schema()
    assert _schema_revision(database_path, schema) == initializer.schema_head(schema)

###############################################################################
def test_postgres_startup_failure_does_not_leak_credentials(monkeypatch) -> None:
    failure = sqlalchemy.exc.OperationalError(
        "postgresql://xreport:secret@127.0.0.1/xreport-test",
        {},
        OSError("connection refused"),
    )
    monkeypatch.setattr(
        initializer,
        "initialize_postgres_database",
        lambda _settings, _schema: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(
        RuntimeError, match="Database startup migration failed"
    ) as exc_info:
        initializer.prepare_database_for_startup(_postgres_settings())

    assert "secret" not in str(exc_info.value).lower()

###############################################################################
@pytest.mark.parametrize(
    ("expected", "reflected"),
    [
        (
            "job_polling_interval >= 0.25 AND job_polling_interval <= 60",
            "job_polling_interval >= 0.25::double precision AND "
            "job_polling_interval <= 60::double precision",
        ),
        (
            "global_seed >= 0 AND global_seed <= 4294967295",
            "global_seed >= 0 AND global_seed <= '4294967295'::numeric",
        ),
        (
            "theme_preference IS NULL OR theme_preference IN ('light', 'dark', 'system')",
            "theme_preference IS NULL OR (theme_preference IN ('light', 'dark', 'system'))",
        ),
    ],
)
def test_postgres_check_constraint_normalization_matches_orm(
    expected: str,
    reflected: str,
) -> None:
    assert initializer._normalize_check_expression(expected) == (
        initializer._normalize_check_expression(reflected)
    )

###############################################################################
def test_postgres_schema_drift_ignores_unique_constraint_backing_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = sqlalchemy.MetaData()
    sqlalchemy.Table(
        "sample",
        metadata,
        sqlalchemy.Column("sample_id", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("name_key", sqlalchemy.String, nullable=False),
        sqlalchemy.UniqueConstraint("name_key", name="uq_sample_name_key"),
    )

    ###############################################################################
    class ReflectedSchema:

        # -------------------------------------------------------------------------
        def get_pk_constraint(self, _table_name: str) -> dict[str, list[str]]:
            return {"constrained_columns": ["sample_id"]}

        # -------------------------------------------------------------------------
        def get_unique_constraints(self, _table_name: str) -> list[dict[str, object]]:
            return [{"name": "uq_sample_name_key", "column_names": ["name_key"]}]

        # -------------------------------------------------------------------------
        def get_foreign_keys(self, _table_name: str) -> list[dict[str, object]]:
            return []

        # -------------------------------------------------------------------------
        def get_indexes(self, _table_name: str) -> list[dict[str, object]]:
            return [
                {
                    "name": "uq_sample_name_key",
                    "column_names": ["name_key"],
                    "unique": True,
                    "duplicates_constraint": "uq_sample_name_key",
                }
            ]

        # -------------------------------------------------------------------------
        def get_check_constraints(self, _table_name: str) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(initializer, "inspect", lambda _connection: ReflectedSchema())
    engine = sqlalchemy.create_engine("sqlite://")
    try:
        with engine.connect() as connection:
            assert initializer._semantic_constraint_diffs(connection, metadata) == []
    finally:
        engine.dispose()






###############################################################################
def test_database_switch_gate_drains_active_requests_and_holds_new_requests() -> None:
    async def scenario() -> None:
        gate = DatabaseRequestGate()
        await gate.enter()

        draining = asyncio.create_task(gate.pause_and_drain())
        await asyncio.sleep(0)
        waiting_request = asyncio.create_task(gate.enter())
        await asyncio.sleep(0)

        assert not draining.done()
        assert not waiting_request.done()

        await gate.leave()
        await draining
        assert not waiting_request.done()

        await gate.resume()
        await waiting_request
        await gate.leave()

    # Playwright's sync API in earlier tests can leave a loop running on this
    # thread, so the scenario runs on a clean thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(asyncio.run, scenario()).result()
