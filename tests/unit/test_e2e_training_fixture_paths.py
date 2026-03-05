from __future__ import annotations

import os
import shutil
import uuid

from tests.e2e import test_training_api


def test_get_checkpoints_root_points_to_backend_resources() -> None:
    expected = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "XREPORT",
            "resources",
            "checkpoints",
        )
    )

    assert test_training_api.get_checkpoints_root() == expected


def test_create_checkpoint_fixture_creates_expected_files() -> None:
    checkpoint_name = f"unit_fixture_{uuid.uuid4().hex}"
    checkpoint_dir = test_training_api.create_checkpoint_fixture(checkpoint_name)

    try:
        assert checkpoint_dir.startswith(test_training_api.get_checkpoints_root())
        assert os.path.isfile(os.path.join(checkpoint_dir, "saved_model.keras"))
        assert os.path.isfile(os.path.join(checkpoint_dir, "nested", "artifact.txt"))
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
