from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from server.domain.validation import CheckpointEvaluationRequest, ValidationRequest
from server.services.jobs import JobExecutionError
import server.services.validation_runs as validation_runs
from server.services.validation_runs import ValidationService
from tests.conftest import run_async_in_thread


###############################################################################
def _service(*, global_seed: int = 123) -> tuple[ValidationService, MagicMock, MagicMock]:
    job_manager = MagicMock()
    job_manager.is_job_running.return_value = False
    job_manager.start_job.return_value = "job-1"
    job_manager.get_job_status.return_value = {
        "job_type": "validation",
        "status": "running",
    }
    checkpoint_repository = MagicMock()
    settings = SimpleNamespace(
        global_settings=SimpleNamespace(seed=global_seed),
        jobs=SimpleNamespace(polling_interval=1.0),
    )
    service = ValidationService(
        job_manager=job_manager,
        server_settings=settings,  # type: ignore[arg-type]
        checkpoint_repository=checkpoint_repository,
    )
    return service, job_manager, checkpoint_repository


###############################################################################
def test_checkpoint_evaluation_persistence_failure_is_terminal(monkeypatch) -> None:
    repository = MagicMock()
    repository.save_checkpoint_evaluation_report.side_effect = RuntimeError(
        "database unavailable"
    )
    monkeypatch.setattr(validation_runs, "ValidationRepository", lambda: repository)

    with pytest.raises(JobExecutionError) as exc_info:
        validation_runs._save_checkpoint_evaluation_report(
            "checkpoint-1",
            ["evaluation_report"],
            {"evaluation_report": {"data_fraction": 1.0}},
            {"loss": 0.25, "accuracy": 0.9},
        )

    assert exc_info.value.code == "persistence_failed"
    assert exc_info.value.phase == "persistence"
    assert exc_info.value.recoverable is True


###############################################################################
def test_validation_service_preserves_explicit_zero_seed() -> None:
    service, job_manager, _ = _service(global_seed=987)
    request = ValidationRequest(
        dataset_name="dataset-1",
        metrics=["text_statistics"],
        sample_size=1.0,
        seed=0,
    )

    run_async_in_thread(service.run_validation(request))

    call = job_manager.start_job.call_args
    assert call.kwargs["kwargs"]["request_data"]["seed"] == 0


###############################################################################
def test_checkpoint_evaluation_uses_global_seed_when_omitted() -> None:
    service, job_manager, checkpoint_repository = _service(global_seed=321)
    checkpoint_repository.get_checkpoint.return_value = SimpleNamespace(
        artifact_complete=True
    )
    job_manager.get_job_status.return_value = {
        "job_type": "checkpoint_evaluation",
        "status": "running",
    }
    request = CheckpointEvaluationRequest(
        checkpoint="checkpoint-1",
        metrics=["evaluation_report"],
        num_samples=10,
    )

    run_async_in_thread(service.evaluate_checkpoint(request))

    call = job_manager.start_job.call_args
    assert call.kwargs["kwargs"]["request_data"]["seed"] == 321
