from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import pandas as pd
import pytest

import server.services.training as training_module
import server.services.validation_runs as validation_module
from server.domain.training import ProcessDatasetRequest
from server.domain.validation import CheckpointEvaluationRequest, ValidationRequest
from server.domain.jobs import JobStartResponse
from server.domain.training import StartTrainingRequest
from server.services.dataset_processing import DatasetProcessingService
from server.services.errors import ConflictError
from server.services.jobs import JobManager
from server.services.preparation import PreparationService
from server.services.training import TrainingRuntime, TrainingService
from server.services.upload import UploadState
from server.services.validation_runs import ValidationService

###############################################################################
def test_concurrent_training_starts_are_atomically_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager()
    service = TrainingService(
        job_manager=manager,
        training_runtime=TrainingRuntime(),
        checkpoint_repository=object(),  # type: ignore[arg-type]
    )

    ###############################################################################
    class DatasetRepositoryStub:

        # -------------------------------------------------------------------------
        def load_training_data(
            self,
            *,
            only_metadata: bool = False,
            dataset_name: str | None = None,
        ) -> object:
            if only_metadata:
                return {"dataset_name": dataset_name}
            return (
                pd.DataFrame([{"text": "synthetic training row"}]),
                pd.DataFrame(),
                {},
            )

    runner_started = Event()
    release_runner = Event()
    check_barrier = Barrier(2)
    check_lock = Lock()
    synchronized_checks = 0
    real_is_job_running = manager.is_job_running

    def blocking_runner(
        configuration: dict[str, object], job_id: str
    ) -> dict[str, str]:
        assert configuration["dataset_name"] == "concurrency-fixture"
        assert job_id
        runner_started.set()
        assert release_runner.wait(timeout=5)
        return {"state": "released"}

    def synchronize_preflight(job_type: str | None = None) -> bool:
        nonlocal synchronized_checks
        result = real_is_job_running(job_type)
        with check_lock:
            synchronized_checks += 1
            wait_for_peer = synchronized_checks <= 2
        if wait_for_peer:
            check_barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(training_module, "DatasetRepository", DatasetRepositoryStub)
    monkeypatch.setattr(training_module, "run_training_job", blocking_runner)
    monkeypatch.setattr(manager, "is_job_running", synchronize_preflight)
    monkeypatch.setattr(
        service,
        "apply_runtime_training_configuration",
        lambda configuration: configuration.update({"polling_interval": 0.25}),
    )
    request = StartTrainingRequest(
        dataset_name="concurrency-fixture",
        epochs=1,
        batch_size=1,
        num_encoders=1,
        num_decoders=1,
        embedding_dims=64,
        attention_heads=1,
        train_temp=1.0,
        freeze_img_encoder=True,
        use_img_augmentation=False,
        shuffle_with_buffer=False,
        shuffle_size=1,
        save_checkpoints=True,
        use_device_GPU=False,
        device_ID=0,
        jit_compile=False,
        jit_backend="eager",
        use_mixed_precision=False,
        dataloader_workers=0,
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=False,
        plot_training_metrics=False,
        use_scheduler=False,
        target_LR=0.001,
        warmup_steps=0,
    )

    def start_training() -> JobStartResponse | ConflictError:
        try:
            return service.start_training(request)
        except ConflictError as exc:
            return exc

    job_id: str | None = None
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(start_training) for _ in range(2)]
            outcomes = [future.result(timeout=5) for future in futures]

        accepted = [item for item in outcomes if isinstance(item, JobStartResponse)]
        rejected = [item for item in outcomes if isinstance(item, ConflictError)]
        assert len(accepted) == 1
        assert len(rejected) == 1
        assert rejected[0].detail == "Training is already in progress"
        job_id = accepted[0].job_id
        assert runner_started.wait(timeout=5)
        assert len(manager.list_jobs(job_type="training", status="running")) == 1
    finally:
        release_runner.set()

    assert job_id is not None
    manager.threads[job_id].join(timeout=5)
    assert not manager.threads[job_id].is_alive()
    assert manager.get_job_status(job_id)["status"] == "completed"  # type: ignore[index]
    assert not manager.is_job_running("training")


###############################################################################
def test_initial_result_is_visible_before_runner_starts() -> None:
    manager = JobManager()
    observed: dict[str, object] = {}

    def runner(job_id: str) -> dict[str, str]:
        status = manager.get_job_status(job_id)
        assert status is not None
        observed.update(status["result"] or {})
        return {"state": "completed"}

    job_id = manager.start_job(
        job_type="training",
        runner=runner,
        initial_result={"worker_phase": "starting", "progress_percent": 0},
    )
    manager.threads[job_id].join(timeout=2)

    assert observed == {"worker_phase": "starting", "progress_percent": 0}

###############################################################################
@pytest.mark.parametrize(
    ("feature", "job_type", "expected_detail"),
    [
        (
            "dataset_processing",
            "dataset_processing",
            "Dataset processing is already in progress",
        ),
        (
            "validation",
            "validation",
            "Validation is already in progress",
        ),
        (
            "checkpoint_evaluation",
            "checkpoint_evaluation",
            "Checkpoint evaluation is already in progress",
        ),
    ],
)
def test_concurrent_feature_job_starts_are_atomically_rejected(
    monkeypatch: pytest.MonkeyPatch,
    feature: str,
    job_type: str,
    expected_detail: str,
) -> None:
    """Every major exclusive feature path must use the atomic job guard."""

    manager = JobManager()
    settings = SimpleNamespace(
        global_settings=SimpleNamespace(seed=42),
        jobs=SimpleNamespace(polling_interval=0.25),
        features=SimpleNamespace(allow_local_filesystem_access=True),
    )
    runner_started = Event()
    release_runner = Event()

    def blocking_runner(*_args: object, **_kwargs: object) -> dict[str, str]:
        runner_started.set()
        assert release_runner.wait(timeout=5)
        return {"state": "released"}

    if feature == "dataset_processing":

        ###############################################################################
        class DatasetRepositoryStub:

            # -------------------------------------------------------------------------
            def load_source_dataset(self, **_kwargs: object) -> pd.DataFrame:
                return pd.DataFrame([{"path": "fixture.png"}])

        processing_service = DatasetProcessingService(
            repository=DatasetRepositoryStub(),
            job_manager=manager,
        )
        monkeypatch.setattr(processing_service, "run", blocking_runner)
        service = PreparationService(
            repository=object(),  # type: ignore[arg-type]
            dataset_repository=DatasetRepositoryStub(),
            processing_service=processing_service,
            job_manager=manager,
            upload_state=UploadState(),
            server_settings=settings,  # type: ignore[arg-type]
        )
        request = ProcessDatasetRequest(
            dataset_name="concurrency-fixture",
            sample_size=1.0,
            validation_size=0.2,
            tokenizer="fixture-tokenizer",
            max_report_size=50,
        )

        def start_feature() -> JobStartResponse:
            return service.process_dataset(request)

    elif feature == "validation":
        monkeypatch.setattr(validation_module, "run_validation_job", blocking_runner)
        service = ValidationService(manager, settings)  # type: ignore[arg-type]
        request = ValidationRequest(
            dataset_name="concurrency-fixture",
            metrics=["text_statistics"],
            sample_size=1.0,
        )

        def start_feature() -> JobStartResponse:
            return asyncio.run(service.run_validation(request))

    else:

        ###############################################################################
        class CheckpointRepositoryStub:

            # -------------------------------------------------------------------------
            def get_checkpoint(self, _name: str) -> SimpleNamespace:
                return SimpleNamespace(artifact_complete=True)

        monkeypatch.setattr(
            validation_module,
            "run_checkpoint_evaluation_job",
            blocking_runner,
        )
        service = ValidationService(
            manager,
            settings,  # type: ignore[arg-type]
            checkpoint_repository=CheckpointRepositoryStub(),  # type: ignore[arg-type]
        )
        request = CheckpointEvaluationRequest(
            checkpoint="concurrency-checkpoint",
            metrics=["evaluation_report"],
            num_samples=1,
        )

        def start_feature() -> JobStartResponse:
            return asyncio.run(service.evaluate_checkpoint(request))

    preflight_barrier = Barrier(2)
    check_lock = Lock()
    synchronized_checks = 0
    real_is_job_running = manager.is_job_running

    def synchronize_preflight(job_type_filter: str | None = None) -> bool:
        nonlocal synchronized_checks
        result = real_is_job_running(job_type_filter)
        with check_lock:
            synchronized_checks += 1
            wait_for_peer = synchronized_checks <= 2
        if wait_for_peer:
            preflight_barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(manager, "is_job_running", synchronize_preflight)
    job_id: str | None = None
    try:
        def invoke_start() -> JobStartResponse | ConflictError:
            try:
                return start_feature()
            except ConflictError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = [
                future.result(timeout=5)
                for future in [executor.submit(invoke_start) for _ in range(2)]
            ]

        accepted = [item for item in outcomes if isinstance(item, JobStartResponse)]
        rejected = [item for item in outcomes if isinstance(item, ConflictError)]
        assert len(accepted) == 1
        assert len(rejected) == 1
        assert rejected[0].detail == expected_detail
        job_id = accepted[0].job_id
        assert runner_started.wait(timeout=5)
    finally:
        release_runner.set()

    assert job_id is not None
    manager.threads[job_id].join(timeout=5)
    assert not manager.threads[job_id].is_alive()
    assert manager.get_job_status(job_id)["status"] == "completed"  # type: ignore[index]
    assert not manager.is_job_running(job_type)

###############################################################################
@pytest.mark.parametrize(
    ("first_job_type", "second_job_type"),
    [
        ("training", "dataset_processing"),
        ("training", "validation"),
        ("training", "checkpoint_evaluation"),
        ("dataset_processing", "validation"),
        ("dataset_processing", "checkpoint_evaluation"),
        ("validation", "checkpoint_evaluation"),
    ],
)
def test_distinct_exclusive_job_types_overlap_and_complete(
    first_job_type: str,
    second_job_type: str,
) -> None:
    """Distinct job types may overlap without leaving a stuck job behind."""

    manager = JobManager()
    release = Event()
    started = {first_job_type: Event(), second_job_type: Event()}
    active_types: set[str] = set()
    active_lock = Lock()
    overlap_observed = Event()

    def make_runner(job_type: str):
        def blocking_runner(*_args: object, **_kwargs: object) -> dict[str, str]:
            with active_lock:
                active_types.add(job_type)
                if len(active_types) >= 2:
                    overlap_observed.set()
            started[job_type].set()
            release.wait(timeout=5)
            with active_lock:
                active_types.discard(job_type)
            return {"state": "released", "job_type": job_type}

        return blocking_runner

    first_job_id = manager.start_job(
        job_type=first_job_type,
        runner=make_runner(first_job_type),
        require_idle=True,
    )
    assert started[first_job_type].wait(timeout=5)

    second_job_id = manager.start_job(
        job_type=second_job_type,
        runner=make_runner(second_job_type),
        require_idle=True,
    )
    try:
        assert started[second_job_type].wait(timeout=5)
        assert overlap_observed.wait(timeout=5)
        assert manager.is_job_running(first_job_type)
        assert manager.is_job_running(second_job_type)
    finally:
        release.set()

    for job_id in (first_job_id, second_job_id):
        manager.threads[job_id].join(timeout=5)
        assert not manager.threads[job_id].is_alive()
        assert manager.get_job_status(job_id)["status"] == "completed"  # type: ignore[index]

    assert not manager.is_job_running(first_job_type)
    assert not manager.is_job_running(second_job_type)
