from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import sqlalchemy
from alembic import command
from alembic.config import Config

import server.repositories.database.engine as database_engine
import server.repositories.database.initializer as initializer
from server.configurations.settings import DatabaseSettings
from server.repositories.schemas import Base
from server.repositories.schemas.models import InferenceReport, InferenceRun

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
def _patch_sqlite_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(initializer, "DATABASE_FILE_PATH", path)
    monkeypatch.setattr(database_engine, "DATABASE_FILE_PATH", path)

###############################################################################
def _alembic_config(path: Path) -> Config:
    config = Config(str(initializer.MIGRATIONS_DIR.parent / "alembic.ini"))
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
def test_sqlite_startup_migrates_to_head_and_is_repeatable(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)

    initializer.initialize_database(_sqlite_settings())
    initializer.prepare_database_for_startup(_sqlite_settings())

    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        tables = set(sqlalchemy.inspect(engine).get_table_names())
        assert set(Base.metadata.tables).issubset(tables)
        with engine.connect() as connection:
            assert (
                connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one()
                == initializer.HEAD_REVISION
            )
    finally:
        engine.dispose()

###############################################################################
def test_nonempty_unversioned_schema_is_rejected_without_implicit_adoption(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    command.upgrade(_alembic_config(database_path), "c1e4f1a7b2d9")
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO datasets (name, name_key, created_at) "
                "VALUES ('Legacy', 'legacy', '2026-01-01 00:00:00')"
            )
            connection.exec_driver_sql("DROP TABLE alembic_version")
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="implicit schema adoption"):
        initializer.initialize_database(_sqlite_settings())

    assert "alembic_version" not in _database_tables(database_path)

###############################################################################
def test_unknown_partial_schema_is_rejected_without_stamping(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE datasets (dataset_id INTEGER PRIMARY KEY, name TEXT)"
            )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="implicit schema adoption"):
        initializer.prepare_database_for_startup(_sqlite_settings())

    assert "alembic_version" not in _database_tables(database_path)

###############################################################################
def test_failed_migration_rolls_back_database_changes(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)

    def fail_upgrade(config: Config, _revision: str) -> None:
        connection = config.attributes["connection"]
        connection.exec_driver_sql("CREATE TABLE transient_failure (id INTEGER)")
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(initializer.command, "upgrade", fail_upgrade)
    with pytest.raises(RuntimeError, match="Unexpected error"):
        initializer.initialize_database(_sqlite_settings())

    assert "transient_failure" not in _database_tables(database_path)
    assert "alembic_version" not in _database_tables(database_path)

###############################################################################
def test_head_schema_drift_blocks_startup(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    initializer.initialize_database(_sqlite_settings())
    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "ALTER TABLE datasets ADD COLUMN unexpected TEXT"
            )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="differs from the current ORM schema"):
        initializer.prepare_database_for_startup(_sqlite_settings())

###############################################################################
def test_sqlite_enforces_foreign_keys_and_cascades_report_deletion(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "database.db"
    _patch_sqlite_path(monkeypatch, database_path)
    initializer.initialize_database(_sqlite_settings())
    database = database_engine.Database(_sqlite_settings())

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
    _patch_sqlite_path(monkeypatch, database_path)
    settings = _sqlite_settings()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(initializer.initialize_database, settings) for _ in range(2)
        ]
        for future in futures:
            future.result()

    engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    try:
        with engine.connect() as connection:
            assert (
                connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one()
                == initializer.HEAD_REVISION
            )
    finally:
        engine.dispose()

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
        lambda _settings: (_ for _ in ()).throw(failure),
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
            "inference_device IN ('auto', 'cpu', 'cuda')",
            "inference_device = ANY (ARRAY['auto'::character varying, "
            "'cpu'::character varying, 'cuda'::character varying]::character varying[])",
        ),
        (
            "global_seed >= 0 AND global_seed <= 4294967295",
            "global_seed >= 0 AND global_seed <= '4294967295'::numeric",
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

    class ReflectedSchema:
        def get_pk_constraint(self, _table_name: str) -> dict[str, list[str]]:
            return {"constrained_columns": ["sample_id"]}

        def get_unique_constraints(self, _table_name: str) -> list[dict[str, object]]:
            return [{"name": "uq_sample_name_key", "column_names": ["name_key"]}]

        def get_foreign_keys(self, _table_name: str) -> list[dict[str, object]]:
            return []

        def get_indexes(self, _table_name: str) -> list[dict[str, object]]:
            return [
                {
                    "name": "uq_sample_name_key",
                    "column_names": ["name_key"],
                    "unique": True,
                    "duplicates_constraint": "uq_sample_name_key",
                }
            ]

        def get_check_constraints(self, _table_name: str) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(initializer, "inspect", lambda _connection: ReflectedSchema())
    engine = sqlalchemy.create_engine("sqlite://")
    try:
        with engine.connect() as connection:
            assert initializer._semantic_constraint_diffs(connection, metadata) == []
    finally:
        engine.dispose()
