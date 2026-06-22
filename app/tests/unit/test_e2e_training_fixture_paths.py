from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from server.common.path import CHECKPOINTS_DIR
from tests.e2e import test_training_api

###############################################################################
def test_get_checkpoints_root_points_to_backend_resources() -> None:
    expected = str(CHECKPOINTS_DIR)

    assert test_training_api.get_checkpoints_root() == expected

###############################################################################
def test_create_checkpoint_fixture_creates_expected_files() -> None:
    checkpoint_name = f"unit_fixture_{uuid.uuid4().hex}"
    checkpoint_dir = test_training_api.create_checkpoint_fixture(checkpoint_name)

    try:
        assert checkpoint_dir.startswith(test_training_api.get_checkpoints_root())
        checkpoint_path = Path(checkpoint_dir)
        assert (checkpoint_path / "saved_model.keras").is_file()
        assert (checkpoint_path / "nested" / "artifact.txt").is_file()
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
