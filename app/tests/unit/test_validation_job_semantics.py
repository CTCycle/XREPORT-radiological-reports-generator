from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pandas as pd
import pytest

from server.configurations import ServerSettings
from server.domain.validation import CheckpointEvaluationRequest, ValidationRequest
from server.repositories.checkpoints import CheckpointRepository
from server.services.jobs import JobExecutionError, JobManager
from server.services.validation_runs import ValidationService
from server.services import validation_runs
from tests.conftest import run_async_in_thread


###############################################################################
class FakeValidationDatasetRepository:
    def __init__(
        self,
        source: pd.DataFrame,
        validated: pd.DataFrame | None = None,
    ) -> None:
        self.source = source
        self.validated = source if validated is None else validated

    def load_source_dataset(self, **_kwargs: object) -> pd.DataFrame:
        return self.source

    def validate_img_paths(self, _dataset: pd.DataFrame) -> pd.DataFrame:
        return self.validated


###############################################################################
def _job_manager(job_type: str) -> JobManager:
    manager = MagicMock(spec=JobManager)
    manager.is_job_running.return_value = False
    manager.start_job.return_value = "job-1"
    manager.get_job_status.return_value = {
        "job_id": "job-1",
        "job_type": job_type,
        "status": "running",
        "progress": 0.0,
        "result": None,
        "error": None,
    }
    return cast(JobManager, manager)


###############################################################################
def _settings(seed: int = 123) -> ServerSettings:
    return cast(
        ServerSettings,
        SimpleNamespace(
            global_settings=SimpleNamespace(seed=seed),
            jobs=SimpleNamespace(polling_interval=2.0),
        ),
    )


###############################################################################
def test_validation_job_fails_when_dataset_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MagicMock(spec=JobManager)
    manager.should_stop.return_value = False
    repository = FakeValidationDatasetRepository(pd.DataFrame())
    monkeypatch.setattr(validation_runs, "get_job_manager", lambda: manager)
    monkeypatch.setattr(validation_runs, "DatasetRepository", lambda: repository)

    with pytest.raises(JobExecutionError) as raised:
        validation_runs.run_validation_job(
            {
                "dataset_name": "missing-dataset",
                "sample_size": 1.0,
                "metrics": ["text_statistics"],
                "seed": 42,
            },
            "job-1",
        )

    assert raised.value.code == "dataset_unavailable"
    assert raised.value.phase == "input_validation"


###############################################################################
def test_validation_job_fails_when_no_image_paths_remain(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MagicMock(spec=JobManager)
    manager.should_stop.return_value = False
    repository = FakeValidationDatasetRepository(
        pd.DataFrame([{"path": "missing.png", "text": "report"}]),
        validated=pd.DataFrame(),
    )
    monkeypatch.setattr(validation_runs, "get_job_manager", lambda: manager)
    monkeypatch.setattr(validation_runs, "DatasetRepository", lambda: repository)

    with pytest.raises(JobExecutionError) as raised:
        validation_runs.run_validation_job(
            {
                "dataset_name": "broken-dataset",
                "sample_size": 1.0,
                "metrics": ["image_statistics"],
                "seed": 42,
            },
            "job-1",
        )

    assert raised.value.code == "dataset_integrity_failed"
    assert raised.value.phase == "input_validation"


###############################################################################
def test_checkpoint_evaluation_persistence_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MagicMock()
    repository.save_checkpoint_evaluation_report.side_effect = RuntimeError(
        "database unavailable"
    )
    monkeypatch.setattr(validation_runs, "ValidationRepository", lambda: repository)

    with pytest.raises(JobExecutionError) as raised:
        validation_runs._save_checkpoint_evaluation_report(
            "checkpoint-1",
            ["evaluation_report"],
            {"evaluation_report": {"data_fraction": 1.0}},
            {"loss": 0.2, "accuracy": 0.8},
        )

    assert raised.value.code == "persistence_failed"
    assert raised.value.phase == "persistence"


###############################################################################
def test_validation_service_preserves_explicit_zero_seed() -> None:
    manager = _job_manager("validation")
    service = ValidationService(manager, _settings())

    run_async_in_thread(
        service.run_validation(
            ValidationRequest(
                dataset_name="dataset-1",
                metrics=["text_statistics"],
                sample_size=1.0,
                seed=0,
            )
        )
    )

    request_data = manager.start_job.call_args.kwargs["kwargs"]["request_data"]
    assert request_data["seed"] == 0


###############################################################################
def test_validation_service_uses_global_seed_when_omitted() -> None:
    manager = _job_manager("validation")
    service = ValidationService(manager, _settings(seed=321))

    run_async_in_thread(
        service.run_validation(
            ValidationRequest(
                dataset_name="dataset-1",
                metrics=["text_statistics"],
                sample_size=1.0,
            )
        )
    )

    request_data = manager.start_job.call_args.kwargs["kwargs"]["request_data"]
    assert request_data["seed"] == 321


###############################################################################
def test_checkpoint_evaluation_uses_global_seed_when_omitted() -> None:
    manager = _job_manager("checkpoint_evaluation")
    checkpoint_repository = MagicMock(spec=CheckpointRepository)
    checkpoint_repository.get_checkpoint.return_value = SimpleNamespace(
        artifact_complete=True
    )
    service = ValidationService(
        manager,
        _settings(seed=654),
        checkpoint_repository=cast(CheckpointRepository, checkpoint_repository),
    )

    run_async_in_thread(
        service.evaluate_checkpoint(
            CheckpointEvaluationRequest(
                checkpoint="checkpoint-1",
                metrics=["evaluation_report"],
                num_samples=10,
            )
        )
    )

    request_data = manager.start_job.call_args.kwargs["kwargs"]["request_data"]
    assert request_data["seed"] == 654
