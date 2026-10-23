from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

import pytest
from sqlalchemy import create_engine

from server.repositories.schemas import Base

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_to_studio.py"
spec = importlib.util.spec_from_file_location("migrate_to_studio", SCRIPT)
assert spec and spec.loader
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def _create_schema(path: Path) -> None:
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    Base.metadata.create_all(engine)
    engine.dispose()


def _legacy(base: Path) -> None:
    base.mkdir(parents=True)
    _create_schema(base / "database.db")
    checkpoint = base / "checkpoints" / "run1"
    (checkpoint / "configuration").mkdir(parents=True)
    (checkpoint / "saved_model.keras").write_bytes(b"weights")
    with sqlite3.connect(base / "database.db") as connection:
        connection.execute(
            "INSERT INTO checkpoints (name, name_key, path, created_at, last_seen_at) "
            "VALUES ('run1','run1',?, '2026-01-01', '2026-01-01')",
            (str(checkpoint),),
        )
        connection.commit()


###############################################################################
def test_dry_run_changes_nothing_and_reports_disk_needs(tmp_path: Path) -> None:
    base = tmp_path / "data"
    _legacy(base)

    result = migration.migrate(base, base / "studio", lambda: None, dry_run=True)

    assert result["rows"]["checkpoints"] == 1
    assert result["checkpoints"] == ["run1"]
    assert result["estimated_bytes"] > 0
    assert not (base / "studio").exists()


###############################################################################
def test_migration_copies_rewrites_paths_is_idempotent_and_keeps_legacy(
    tmp_path: Path,
) -> None:
    base = tmp_path / "data"
    studio = base / "studio"
    _legacy(base)
    legacy_before = (base / "database.db").read_bytes()

    manifest = migration.migrate(
        base, studio, lambda: _create_schema(studio / "database.db")
    )

    assert manifest["rows"]["checkpoints"] == 1
    assert (studio / "checkpoints" / "run1" / "saved_model.keras").read_bytes() == b"weights"
    with sqlite3.connect(studio / "database.db") as connection:
        stored = connection.execute("SELECT path FROM checkpoints").fetchone()[0]
    assert Path(stored) == (studio / "checkpoints" / "run1").resolve()
    assert (base / "checkpoints" / "run1" / "saved_model.keras").is_file()
    assert (base / "database.db").read_bytes() == legacy_before
    assert Path(manifest["backup"]).is_file()
    assert json.loads((studio / "migration-manifest.json").read_text())["rows"]

    again = migration.migrate(base, studio, lambda: pytest.fail("must not reinitialize"))
    assert again["already_migrated"] is True


###############################################################################
def test_refuses_to_merge_into_a_studio_that_already_has_data(tmp_path: Path) -> None:
    base = tmp_path / "data"
    studio = base / "studio"
    _legacy(base)
    studio.mkdir()
    _create_schema(studio / "database.db")
    with sqlite3.connect(studio / "database.db") as connection:
        connection.execute(
            "INSERT INTO checkpoints (name, name_key, path, created_at, last_seen_at) "
            "VALUES ('other','other','x', '2026-01-01', '2026-01-01')"
        )
        connection.commit()

    with pytest.raises(migration.MigrationError, match="already contains data"):
        migration.migrate(base, studio, lambda: None)

    assert not (studio / "migration-manifest.json").exists()
    assert not (studio / "checkpoints" / "run1").exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_postgresql_source_refuses_stale_sqlite_before_target_initialization(
    tmp_path: Path, dry_run: bool,
) -> None:
    base = tmp_path / "data"
    _legacy(base)
    (base / "database-config.json").write_text(json.dumps({
        "schema_version": 1, "backend": "postgresql", "host": "localhost",
        "database_name": "xreport", "username": "user",
    }))
    with pytest.raises(migration.MigrationError, match="Only the embedded SQLite"):
        migration.migrate(
            base, base / "studio", lambda: pytest.fail("must not initialize"),
            dry_run=dry_run,
        )
    assert not (base / "studio").exists()


def test_migration_uses_configured_sqlite_source(tmp_path: Path) -> None:
    base = tmp_path / "data"
    _legacy(base)
    database = base / "custom.db"
    shutil.copyfile(base / "database.db", database)
    with sqlite3.connect(base / "database.db") as connection:
        connection.execute("DELETE FROM checkpoints")
    (base / "database-config.json").write_text(json.dumps({
        "schema_version": 1, "backend": "sqlite", "sqlite_path": str(database),
    }))
    studio = base / "studio"
    result = migration.migrate(base, studio, lambda: _create_schema(studio / "database.db"))
    assert result["rows"]["checkpoints"] == 1
