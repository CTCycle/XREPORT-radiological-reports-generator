from __future__ import annotations

import ctypes
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from xreport_shared.common.path import DATABASE_FILE_PATH, RUNTIME_LAYOUT
from xreport_shared.common.runtime_layout import (
    STORAGE_ENTRIES,
    read_storage_pointer,
    write_storage_pointer,
)
from xreport_shared.common.utils.logger import logger
from xreport_shared.domain.jobs import JobStartResponse
from xreport_inference.domain.storage_settings import (
    StorageFolder,
    StorageMode,
    StorageStatus,
    StorageTargetState,
    StorageValidationResponse,
)
from xreport_shared.services.errors import BadRequestError, ConflictError
from xreport_shared.services.jobs import JobExecutionError, JobManager, get_job_manager

JOB_TYPE = "storage_migration"
STAGING_PREFIX = ".xreport-migration-"
COPY_CHUNK_BYTES = 8 * 1024 * 1024
FREE_SPACE_MARGIN = 1.1
DATABASE_ENTRY = "database.db"
DATABASE_CONFIG_ENTRY = "database-config.json"
DRIVE_WARNINGS = {2: "removable drive", 4: "network drive"}


###############################################################################
def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))

###############################################################################
def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True

###############################################################################
def _entry_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

###############################################################################
def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate

###############################################################################
def _drive_warning(path: Path) -> str | None:
    if os.name != "nt" or not path.drive:
        return None
    kind = ctypes.windll.kernel32.GetDriveTypeW(f"{path.drive}\\")  # type: ignore[attr-defined]
    label = DRIVE_WARNINGS.get(kind)
    if label is None:
        return None
    return f"The location is on a {label}; models and the database may be slow or unavailable."

###############################################################################
def _entries_present(root: Path) -> list[Path]:
    return [root / name for name in STORAGE_ENTRIES if (root / name).exists()]

###############################################################################
def _rewrite_paths(staged: Path, old_root: Path, new_root: Path) -> None:
    """Point the copied checkpoint registry and database config at the new root."""
    database = staged / DATABASE_ENTRY
    if database.is_file():
        connection = sqlite3.connect(database)
        try:
            has_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='checkpoints'"
            ).fetchone()
            if has_table:
                rows = connection.execute(
                    "SELECT checkpoint_id, path FROM checkpoints"
                ).fetchall()
                for checkpoint_id, path in rows:
                    stored = Path(path)
                    if _is_within(stored, old_root):
                        connection.execute(
                            "UPDATE checkpoints SET path = ? WHERE checkpoint_id = ?",
                            (str(new_root / stored.relative_to(old_root)), checkpoint_id),
                        )
            connection.commit()
        finally:
            connection.close()
    config = staged / DATABASE_CONFIG_ENTRY
    if config.is_file():
        payload = json.loads(config.read_text(encoding="utf-8"))
        sqlite_path = payload.get("sqlite_path")
        if isinstance(sqlite_path, str) and _same_path(
            Path(sqlite_path), old_root / DATABASE_ENTRY
        ):
            payload["sqlite_path"] = str(new_root / DATABASE_ENTRY)
            config.write_text(json.dumps(payload, indent=2), encoding="utf-8")

###############################################################################
def _backup_database(source: Path, destination: Path) -> None:
    origin = sqlite3.connect(source)
    copy = sqlite3.connect(destination)
    try:
        origin.backup(copy)
    finally:
        copy.close()
        origin.close()

###############################################################################
def run_storage_migration(
    source_root: str,
    target_root: str,
    mode: StorageMode,
    job_id: str,
    *,
    job_manager: JobManager,
    bootstrap_dir: str,
    default_root: str,
) -> dict[str, Any]:
    """Copy, verify, then switch; the previous root is never modified or removed."""
    source, target = Path(source_root), Path(target_root)
    staging = target / f"{STAGING_PREFIX}{job_id}"
    copied = 0

    def report(message: str, fraction: float) -> None:
        job_manager.update_progress(job_id, round(min(fraction, 1.0) * 95, 1))
        job_manager.update_result(
            job_id, {"lifecycle": {"phase": "migrating", "message": message}}
        )

    try:
        if mode == "copy":
            entries = _entries_present(source)
            total = max(1, sum(_entry_size(entry) for entry in entries))
            target.mkdir(parents=True, exist_ok=True)
            staging.mkdir()
            for entry in entries:
                destination = staging / entry.name
                if entry.name == DATABASE_ENTRY:
                    report("Copying the application database", copied / total)
                    _backup_database(entry, destination)
                    copied += entry.stat().st_size
                    continue
                if entry.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                files = (
                    [entry]
                    if entry.is_file()
                    else [item for item in entry.rglob("*") if item.is_file()]
                )
                for file in files:
                    if job_manager.should_stop(job_id):
                        raise InterruptedError
                    out = destination if entry.is_file() else destination / file.relative_to(entry)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with file.open("rb") as reader, out.open("wb") as writer:
                        while chunk := reader.read(COPY_CHUNK_BYTES):
                            writer.write(chunk)
                            copied += len(chunk)
                            report(f"Copying {entry.name}", copied / total)
                    if out.stat().st_size != file.stat().st_size:
                        raise JobExecutionError(
                            f"Copy verification failed for {file.name}",
                            code="storage_verification_failed",
                            phase="verifying",
                            recoverable=True,
                        )
            report("Updating stored paths", 0.97)
            _rewrite_paths(staging, source, target)
            for entry in staging.iterdir():
                os.replace(entry, target / entry.name)
        write_storage_pointer(
            Path(bootstrap_dir),
            None if _same_path(target, Path(default_root)) else target,
        )
        job_manager.update_progress(job_id, 100)
        return {
            "target_root": str(target),
            "copied_bytes": copied,
            "restart_required": True,
        }
    except InterruptedError:
        return {}
    finally:
        shutil.rmtree(staging, ignore_errors=True)

###############################################################################
class StorageSettingsService:
    """Chooses where models, application data and logs live (applied on restart)."""

    # -------------------------------------------------------------------------
    def __init__(self, job_manager: JobManager | None = None) -> None:
        self.layout = RUNTIME_LAYOUT
        self.job_manager = job_manager or get_job_manager()

    # -------------------------------------------------------------------------
    @property
    def active_root(self) -> Path:
        return self.layout.resources_root

    # -------------------------------------------------------------------------
    @property
    def default_root(self) -> Path:
        if self.layout.packaged:
            return self.layout.data_root
        return self.layout.runtime_root / "data"

    # -------------------------------------------------------------------------
    @property
    def locked(self) -> bool:
        return self.layout.storage_source in {"environment", "env_file"}

    # -------------------------------------------------------------------------
    def get_status(self) -> StorageStatus:
        pointer = read_storage_pointer(self.layout.bootstrap_dir)
        pending = Path(pointer) if pointer else self.default_root
        restart_required = not self.locked and not _same_path(pending, self.active_root)
        return StorageStatus(
            active_root=str(self.active_root),
            default_root=str(self.default_root),
            pending_root=str(pending) if restart_required else None,
            source=self.layout.storage_source,  # type: ignore[arg-type]
            locked=self.locked,
            restart_required=restart_required,
            folders=[
                StorageFolder(name=entry.name, path=str(entry), size_bytes=_entry_size(entry))
                for entry in _entries_present(self.active_root)
            ],
        )

    # -------------------------------------------------------------------------
    def validate(self, raw_path: str) -> StorageValidationResponse:
        errors: list[str] = []
        warnings: list[str] = []
        candidate = Path(raw_path.strip()).expanduser()
        if not candidate.is_absolute():
            return StorageValidationResponse(
                valid=False,
                resolved_path=raw_path,
                errors=["Enter an absolute folder path."],
            )
        resolved = candidate.resolve()
        if _same_path(resolved, self.active_root):
            errors.append("This is already the active storage location.")
        elif _is_within(resolved, self.active_root) or _is_within(self.active_root, resolved):
            errors.append("The location cannot be inside the current storage folder or contain it.")
        forbidden = (
            self.layout.runtime_root
            if self.layout.packaged
            else self.layout.runtime_root / "app"
        )
        if _is_within(resolved, forbidden):
            errors.append("The location cannot be inside the application files.")
        if resolved.exists() and not resolved.is_dir():
            errors.append("The path exists but is not a folder.")
        target_state: StorageTargetState = "new"
        required = 0
        free: int | None = None
        if not errors:
            anchor = _nearest_existing(resolved)
            try:
                with tempfile.TemporaryFile(dir=anchor):
                    pass
            except OSError:
                errors.append("The location is not writable.")
            if not errors and resolved.is_dir():
                if _entries_present(resolved):
                    target_state = "existing"
                elif any(resolved.iterdir()):
                    errors.append(
                        "The folder is not empty and holds no XREPORT data. Choose an empty folder."
                    )
                else:
                    target_state = "empty"
            if not errors:
                required = sum(_entry_size(entry) for entry in _entries_present(self.active_root))
                free = shutil.disk_usage(anchor).free
                if target_state != "existing":
                    if free < required * FREE_SPACE_MARGIN:
                        errors.append("There is not enough free space for the current data.")
                    if not self._database_relocatable():
                        errors.append(
                            "The application database is not the default file in the storage folder, so its data cannot be moved automatically."
                        )
                else:
                    warnings.append(
                        "This folder already contains XREPORT data. It will be used as-is; current data is not copied or overwritten."
                    )
                if warning := _drive_warning(resolved):
                    warnings.append(warning)
        return StorageValidationResponse(
            valid=not errors,
            resolved_path=str(resolved),
            target_state=None if errors else target_state,
            required_bytes=required,
            free_bytes=free,
            errors=errors,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    def apply(self, raw_path: str, mode: StorageMode) -> JobStartResponse:
        if self.locked:
            raise ConflictError(
                detail="The storage location is set by XREPORT_RESOURCES_DIR in the environment or settings/.env; change it there."
            )
        if self.job_manager.is_job_running():
            raise ConflictError(
                detail="Wait for active background jobs to finish before moving storage."
            )
        validation = self.validate(raw_path)
        if not validation.valid:
            raise BadRequestError(detail=" ".join(validation.errors))
        if (mode == "use_existing") != (validation.target_state == "existing"):
            raise BadRequestError(
                detail="Existing XREPORT data must be used as-is; copying is only for new or empty folders."
            )
        job_id = self.job_manager.start_job(
            job_type=JOB_TYPE,
            runner=run_storage_migration,
            require_idle=True,
            kwargs={
                "source_root": str(self.active_root),
                "target_root": validation.resolved_path,
                "mode": mode,
                "job_manager": self.job_manager,
                "bootstrap_dir": str(self.layout.bootstrap_dir),
                "default_root": str(self.default_root),
            },
        )
        logger.info("Started storage migration job %s to %s", job_id, validation.resolved_path)
        return JobStartResponse(
            job_id=job_id,
            job_type=JOB_TYPE,
            status="running",
            message="Storage location change started",
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _database_relocatable() -> bool:
        from xreport_shared.configurations.startup import get_database_settings

        settings = get_database_settings()
        return settings.backend == "sqlite" and (
            settings.sqlite_path is None
            or _same_path(Path(settings.sqlite_path).resolve(), DATABASE_FILE_PATH.resolve())
        )
