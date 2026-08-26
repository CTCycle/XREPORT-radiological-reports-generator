"""Cross-layer checks for destructive and resume-sensitive training API behavior."""

import shutil
import uuid
from pathlib import Path

from playwright.sync_api import APIRequestContext

from server.common.path import CHECKPOINTS_DIR


###############################################################################
def _create_checkpoint_fixture(name: str) -> Path:
    checkpoint_dir = Path(CHECKPOINTS_DIR) / name
    (checkpoint_dir / "nested").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "saved_model.keras").write_text("placeholder", encoding="utf-8")
    (checkpoint_dir / "nested" / "artifact.txt").write_text("nested placeholder", encoding="utf-8")
    return checkpoint_dir


###############################################################################
def test_delete_checkpoint_removes_the_entire_checkpoint_directory(
    api_context: APIRequestContext,
) -> None:
    checkpoint_name = f"e2e_delete_{uuid.uuid4().hex}"
    checkpoint_dir = _create_checkpoint_fixture(checkpoint_name)

    try:
        response = api_context.delete(f"/api/training/checkpoints/{checkpoint_name}")
        assert response.ok, f"Expected 200, got {response.status}"
        assert not checkpoint_dir.exists()
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


###############################################################################
def test_delete_checkpoint_rejects_path_traversal(api_context: APIRequestContext) -> None:
    response = api_context.delete("/api/training/checkpoints/%2e%2e%2f%2e%2e%2f")

    assert response.status == 400


###############################################################################
def test_resume_rejects_unknown_checkpoint(api_context: APIRequestContext) -> None:
    response = api_context.post(
        "/api/training/resume",
        data={"checkpoint": "non_existent_checkpoint_xyz", "additional_epochs": 1},
    )

    assert response.status in {400, 404}
