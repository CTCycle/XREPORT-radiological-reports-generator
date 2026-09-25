from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pandas as pd
import pytest

import server.services.training as training_module
from server.domain.jobs import JobStartResponse
from server.domain.training import StartTrainingRequest
from server.services.errors import ConflictError
from server.services.jobs import JobManager
from server.services.training import TrainingRuntime, TrainingService


def test_concurrent_training_starts_are_atomically_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager()
    service = TrainingService(
        job_manager=manager,
        training_runtime=TrainingRuntime(),
        checkpoint_repository=object(),  # type: ignore[arg-type]
    )

    class DatasetRepositoryStub:
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
