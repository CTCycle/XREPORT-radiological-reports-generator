"""Copy existing datasets, checkpoints and evaluations into XREPORT Studio.

Non-destructive and repeatable: the legacy database and folders are never
modified or deleted, a backup of the legacy database is taken first, and a
manifest in the Studio folder marks a completed migration. To roll back, stop
Studio and delete the ``studio`` folder inside the data directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys

APP_DIR = Path(__file__).resolve().parents[1]

MANIFEST_NAME = "migration-manifest.json"
# Parents before children so foreign keys are satisfied while copying.
STUDIO_TABLES = (
    "datasets",
    "dataset_versions",
    "dataset_records",
    "processing_runs",
    "training_samples",
    "validation_runs",
    "checkpoints",
    "checkpoint_evaluations",
)
TEMP_PREFIX = ".migrating-"


###############################################################################
class MigrationError(RuntimeError):
    """Raised when the migration cannot proceed safely."""


###############################################################################
def _folder_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


###############################################################################
def _table_columns(connection: sqlite3.Connection, schema: str, table: str) -> list[str]:
    rows = connection.execute(f'PRAGMA {schema}.table_info("{table}")').fetchall()
    return [row[1] for row in rows]


###############################################################################
def plan(base_dir: Path) -> dict:
    """Describe what would be copied without touching anything."""
    database = base_dir / "database.db"
    if not database.is_file():
        raise MigrationError(f"No legacy database found at {database}")
    counts: dict[str, int] = {}
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as connection:
        for table in STUDIO_TABLES:
            try:
                counts[table] = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = 0
        checkpoints = [
            Path(row[0]).name
            for row in connection.execute("SELECT path FROM checkpoints")
        ] if counts["checkpoints"] else []
    checkpoint_bytes = sum(
        _folder_size(base_dir / "checkpoints" / name)
        for name in checkpoints
        if (base_dir / "checkpoints" / name).is_dir()
    )
    return {
        "rows": counts,
        "checkpoints": checkpoints,
        "estimated_bytes": checkpoint_bytes + database.stat().st_size,
    }


###############################################################################
def _assert_target_is_empty(connection: sqlite3.Connection) -> None:
    for table in STUDIO_TABLES:
        if connection.execute(f'SELECT 1 FROM main."{table}" LIMIT 1').fetchone():
            raise MigrationError(
                "Studio already contains data; refusing to merge it with legacy records."
            )


###############################################################################
def _copy_rows(
    connection: sqlite3.Connection, studio_dir: Path
) -> dict[str, int]:
    copied: dict[str, int] = {}
    for table in STUDIO_TABLES:
        target_columns = _table_columns(connection, "main", table)
        shared = [c for c in target_columns if c in _table_columns(connection, "legacy", table)]
        names = ", ".join(f'"{column}"' for column in shared)
        connection.execute(
            f'INSERT INTO main."{table}" ({names}) SELECT {names} FROM legacy."{table}"'
        )
        copied[table] = connection.execute(
            f'SELECT COUNT(*) FROM main."{table}"'
        ).fetchone()[0]
    for checkpoint_id, path in connection.execute(
        "SELECT checkpoint_id, path FROM main.checkpoints"
    ).fetchall():
        rewritten = (studio_dir / "checkpoints" / Path(path).name).resolve()
        connection.execute(
            "UPDATE main.checkpoints SET path = ? WHERE checkpoint_id = ?",
            (str(rewritten), checkpoint_id),
        )
    return copied


###############################################################################
def _copy_checkpoint_folders(
    names: list[str], base_dir: Path, studio_dir: Path
) -> list[str]:
    target_root = studio_dir / "checkpoints"
    target_root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    try:
        for name in names:
            source = base_dir / "checkpoints" / name
            destination = target_root / name
            if not source.is_dir() or destination.exists():
                continue
            temporary = target_root / f"{TEMP_PREFIX}{name}"
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.copytree(source, temporary)
            temporary.rename(destination)
            created.append(name)
    except BaseException:
        for name in created:
            shutil.rmtree(target_root / name, ignore_errors=True)
        raise
    finally:
        for leftover in target_root.glob(f"{TEMP_PREFIX}*"):
            shutil.rmtree(leftover, ignore_errors=True)
    return created


###############################################################################
def migrate(
    base_dir: Path,
    studio_dir: Path,
    initialize_target: Callable[[], None],
    dry_run: bool = False,
) -> dict:
    manifest_path = studio_dir / MANIFEST_NAME
    if manifest_path.is_file():
        return {**json.loads(manifest_path.read_text(encoding="utf-8")), "already_migrated": True}
    summary = plan(base_dir)
    if dry_run:
        return {**summary, "dry_run": True}

    backup_dir = base_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"database-pre-studio-{stamp}.db"
    with sqlite3.connect(f"file:{(base_dir / 'database.db').as_posix()}?mode=ro", uri=True) as source, sqlite3.connect(backup) as target:
        source.backup(target)

    studio_dir.mkdir(parents=True, exist_ok=True)
    initialize_target()
    target_path = studio_dir / "database.db"
    connection = sqlite3.connect(
        f"file:{target_path.as_posix()}?mode=rw", uri=True, isolation_level=None
    )
    created_folders: list[str] = []
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "ATTACH DATABASE ? AS legacy",
            (f"file:{(base_dir / 'database.db').as_posix()}?mode=ro",),
        )
        connection.execute("BEGIN IMMEDIATE")
        _assert_target_is_empty(connection)
        copied = _copy_rows(connection, studio_dir)
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise MigrationError(f"Foreign key violations after copy: {len(violations)}")
        created_folders = _copy_checkpoint_folders(summary["checkpoints"], base_dir, studio_dir)
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        for name in created_folders:
            shutil.rmtree(studio_dir / "checkpoints" / name, ignore_errors=True)
        raise
    finally:
        connection.close()

    manifest = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(base_dir),
        "backup": str(backup),
        "rows": copied,
        "checkpoint_folders": created_folders,
        "rollback": "Stop Studio and delete the studio folder; legacy data was not modified.",
    }
    temporary = studio_dir / f".{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


###############################################################################
def _format_bytes(value: int) -> str:
    return f"{value / 1024**2:.1f} MiB"


###############################################################################
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only")
    arguments = parser.parse_args()

    import os

    os.environ["XREPORT_APP"] = "studio"
    sys.path.insert(0, str(APP_DIR))
    from server.common.path import RUNTIME_LAYOUT
    from server.configurations import get_database_settings
    from server.repositories.database.initializer import initialize_database

    base_dir = RUNTIME_LAYOUT.shared_resources_root or RUNTIME_LAYOUT.resources_root
    studio_dir = RUNTIME_LAYOUT.resources_root
    settings = get_database_settings()
    if settings.backend != "sqlite":
        print("Only the embedded SQLite database can be migrated automatically.")
        return 2
    try:
        result = migrate(
            base_dir,
            studio_dir,
            lambda: initialize_database(settings),
            dry_run=arguments.dry_run,
        )
    except MigrationError as exc:
        print(f"Migration stopped: {exc}")
        return 1
    if result.get("already_migrated"):
        print(f"Studio was already migrated on {result.get('completed_utc')}; nothing to do.")
        return 0
    print(json.dumps(result.get("rows"), indent=2))
    if result.get("dry_run"):
        print(
            f"Dry run: would copy {len(result['checkpoints'])} checkpoint(s); "
            f"extra disk needed about {_format_bytes(result['estimated_bytes'])}."
        )
    else:
        print(f"Migration complete. Backup: {result['backup']}")
        print(result["rollback"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
