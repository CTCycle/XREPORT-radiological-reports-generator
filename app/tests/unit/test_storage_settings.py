from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from server.common.runtime_layout import RuntimeLayout, read_storage_pointer
from server.services import storage_settings as module
from server.services.errors import ConflictError
from server.services.jobs import JobManager, JobState
from server.services.storage_settings import StorageSettingsService, run_storage_migration


###############################################################################
def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str = "default") -> StorageSettingsService:
    runtime_root = tmp_path / "repo"
    active = runtime_root / "data"
    (active / "logs").mkdir(parents=True)
    (active / "logs" / "app.log").write_text("log line")
    (active / "models").mkdir()
    (active / "models" / "weights.bin").write_bytes(b"weights")
    (active / "checkpoints" / "run-1").mkdir(parents=True)
    (active / "checkpoints" / "run-1" / "model.keras").write_bytes(b"checkpoint")
    connection = sqlite3.connect(active / "database.db")
    connection.execute("CREATE TABLE checkpoints (checkpoint_id INTEGER PRIMARY KEY, path TEXT)")
    connection.execute(
        "INSERT INTO checkpoints (path) VALUES (?)", (str(active / "checkpoints" / "run-1"),)
    )
    connection.commit()
    connection.close()
    (active / "database-config.json").write_text(
        json.dumps({"backend": "sqlite", "sqlite_path": str(active / "database.db")})
    )
    layout = RuntimeLayout(
        "source", runtime_root, active, active, runtime_root, None, None, "xreport", active, source
    )
    monkeypatch.setattr(module, "RUNTIME_LAYOUT", layout)
    monkeypatch.setattr(StorageSettingsService, "_database_relocatable", staticmethod(lambda: True))
    return StorageSettingsService(JobManager())


###############################################################################
def test_copy_migration_rewrites_paths_and_leaves_the_old_root_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    target = tmp_path / "elsewhere" / "xreport-data"
    assert service.validate(str(target)).target_state == "new"
    manager = JobManager()
    manager.jobs["job"] = JobState(job_id="job", job_type="storage_migration", status="running")

    result = run_storage_migration(
        str(service.active_root),
        str(target),
        "copy",
        "job",
        job_manager=manager,
        bootstrap_dir=str(service.layout.bootstrap_dir),
        default_root=str(service.default_root),
    )

    assert result["restart_required"] is True
    assert (target / "models" / "weights.bin").read_bytes() == b"weights"
    assert (target / "logs" / "app.log").read_text() == "log line"
    assert not list(target.glob(".xreport-migration-*"))
    with sqlite3.connect(target / "database.db") as connection:
        assert connection.execute("SELECT path FROM checkpoints").fetchone()[0] == str(
            target / "checkpoints" / "run-1"
        )
    assert json.loads((target / "database-config.json").read_text())["sqlite_path"] == str(
        target / "database.db"
    )
    assert (service.active_root / "models" / "weights.bin").exists()
    with sqlite3.connect(service.active_root / "database.db") as connection:
        assert str(service.active_root) in connection.execute("SELECT path FROM checkpoints").fetchone()[0]
    assert read_storage_pointer(service.layout.bootstrap_dir) == str(target)
    status = service.get_status()
    assert status.restart_required and status.pending_root == str(target)


###############################################################################
def test_validation_rejects_unsafe_locations_and_requires_existing_data_to_be_used_as_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("keep")
    existing = tmp_path / "existing"
    (existing / "models").mkdir(parents=True)

    assert not service.validate("relative/path").valid
    assert not service.validate(str(service.active_root / "nested")).valid
    assert not service.validate(str(service.layout.runtime_root / "app" / "data")).valid
    assert not service.validate(str(unrelated)).valid
    validation = service.validate(str(existing))
    assert validation.valid and validation.target_state == "existing"
    with pytest.raises(Exception, match="used as-is"):
        service.apply(str(existing), "copy")
    assert (unrelated / "notes.txt").read_text() == "keep"


###############################################################################
def test_environment_configured_storage_cannot_be_changed_from_the_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, monkeypatch, source="environment")

    assert service.get_status().locked is True
    with pytest.raises(ConflictError):
        service.apply(str(tmp_path / "new-root"), "copy")
