"""Cross-layer checks for destructive and resume-sensitive training API behavior."""

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import APIRequestContext
from sqlalchemy import delete

from server.common.path import CHECKPOINTS_DIR
from server.repositories.checkpoints import CheckpointRepository
from server.repositories.schemas import CheckpointEvaluation


###############################################################################
def _create_checkpoint_fixture(name: str) -> Path:
    checkpoint_dir = Path(CHECKPOINTS_DIR) / name
    (checkpoint_dir / "nested").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "saved_model.keras").write_text("placeholder", encoding="utf-8")
    configuration_dir = checkpoint_dir / "configuration"
    configuration_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("configuration.json", "metadata.json"):
        (configuration_dir / filename).write_text("{}", encoding="utf-8")
    (configuration_dir / "session_history.json").write_text(
        '{"epochs": 1, "history": {"loss": [0.5], "val_loss": [0.6]}}',
        encoding="utf-8",
    )
    (checkpoint_dir / "nested" / "artifact.txt").write_text(
        "nested placeholder", encoding="utf-8"
    )
    CheckpointRepository().register_completed_checkpoint(name, checkpoint_dir)
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
        listed = api_context.get("/api/training/checkpoints")
        assert listed.ok, listed.text()
        assert all(
            checkpoint["name"] != checkpoint_name
            for checkpoint in listed.json()["checkpoints"]
        )
    finally:
        repository = CheckpointRepository()
        if repository.get_checkpoint(checkpoint_name) is not None:
            repository.delete_checkpoint(checkpoint_name)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


###############################################################################
def test_delete_checkpoint_rejects_path_traversal(
    api_context: APIRequestContext,
) -> None:
    response = api_context.delete("/api/training/checkpoints/%2e%2e%2f%2e%2e%2f")

    assert response.status == 400


def test_delete_checkpoint_rejects_referenced_checkpoint(
    api_context: APIRequestContext,
) -> None:
    checkpoint_name = f"e2e_referenced_{uuid.uuid4().hex}"
    checkpoint_dir = _create_checkpoint_fixture(checkpoint_name)
    repository = CheckpointRepository()
    record = repository.get_checkpoint(checkpoint_name)
    assert record is not None

    try:
        with repository.database.transaction() as session:
            session.add(
                CheckpointEvaluation(
                    checkpoint_id=record.checkpoint_id,
                    executed_at=datetime.now(timezone.utc),
                    metrics_json=[],
                    metric_configs_json=[],
                    results_json=[],
                )
            )

        response = api_context.delete(f"/api/training/checkpoints/{checkpoint_name}")
        assert response.status == 409, response.text()
        assert checkpoint_dir.is_dir()
        assert repository.get_checkpoint(checkpoint_name) is not None
    finally:
        with repository.database.transaction() as session:
            session.execute(
                delete(CheckpointEvaluation).where(
                    CheckpointEvaluation.checkpoint_id == record.checkpoint_id
                )
            )
        if repository.get_checkpoint(checkpoint_name) is not None:
            repository.delete_checkpoint(checkpoint_name)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


###############################################################################
def test_resume_rejects_unknown_checkpoint(api_context: APIRequestContext) -> None:
    response = api_context.post(
        "/api/training/resume",
        data={"checkpoint": "non_existent_checkpoint_xyz", "additional_epochs": 1},
    )

    assert response.status in {400, 404}
