from __future__ import annotations

import json
import tempfile
from pathlib import Path

from server.repositories.serialization.model import ModelSerializer


def test_scan_checkpoints_requires_complete_serialized_checkpoint(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
        checkpoints_dir = Path(temporary_directory)
        incomplete = checkpoints_dir / "incomplete"
        incomplete.mkdir()
        (incomplete / "saved_model.keras").write_text("placeholder")

        complete = checkpoints_dir / "complete"
        (complete / "configuration").mkdir(parents=True)
        (complete / "saved_model.keras").write_text("placeholder")
        for name in ("configuration.json", "metadata.json", "session_history.json"):
            (complete / "configuration" / name).write_text(json.dumps({}))

        monkeypatch.setattr(
            "server.repositories.serialization.model.CHECKPOINTS_DIR",
            checkpoints_dir,
        )

        assert ModelSerializer().scan_checkpoints_folder() == ["complete"]
