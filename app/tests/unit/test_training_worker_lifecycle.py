from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
# ruff: noqa: E402
# Select the backend before importing the training runtime.
import pickle
import keras
import pandas as pd
import torch
import pytest
import queue
from pathlib import Path
from PIL import Image
from server.models.device import DeviceConfig
from server.models.training.dataloader import XRAYDataLoader
from server.repositories.serialization.model import ModelSerializer
from unittest.mock import Mock
from server.services import training as training_module
from server.services.jobs import JobExecutionError
from threading import Event
from server.models.callbacks import TrainingProgressCallback
from server.services.training_worker import ProcessWorker, QueueProgressReporter, WorkerChannels, keep_saved_epoch, process_target
















###############################################################################
def test_cpu_choice_keeps_weights_on_cpu_when_cuda_is_available() -> None:
    # Keras' torch backend defaults to CUDA, so the CPU choice only holds
    # inside the device scope the worker opens.
    device = DeviceConfig({"use_device_GPU": False}).set_device()
    with keras.device(device):
        layer = keras.layers.Dense(2)
        layer.build((None, 3))

    assert device == "cpu"
    assert layer.kernel.value.device.type == "cpu"

###############################################################################
def test_cancelled_run_keeps_saved_epoch_and_drops_empty_folder(
    tmp_path: Path,
) -> None:
    saved = tmp_path / "saved"
    (saved / "configuration").mkdir(parents=True)
    (saved / "saved_model.keras").write_bytes(b"weights")
    ModelSerializer().save_training_configuration(
        saved,
        {"history": {"loss": [2.0, 1.5], "val_loss": [2.1, 1.7]}, "epochs": 2},
        {"epochs": 5},
        {"max_report_size": 8},
    )
    empty = tmp_path / "empty"
    (empty / "configuration").mkdir(parents=True)

    assert keep_saved_epoch(str(saved), created_by_run=True) == {
        "epochs": 2,
        "final_loss": 1.5,
        "final_val_loss": 1.7,
        "checkpoint_path": str(saved),
    }
    assert keep_saved_epoch(str(empty), created_by_run=True) == {}
    assert saved.is_dir()
    assert not empty.exists()

###############################################################################
def test_xray_dataloader_construction_does_not_load_tokenizer() -> None:
    for _ in range(20):
        XRAYDataLoader(
            {
                "batch_size": 2,
                "inference_batch_size": 2,
                "dataloader_workers": 0,
            }
        )








###############################################################################
class _Clock:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.value = 0.0

    # -------------------------------------------------------------------------
    def __call__(self) -> float:
        current = self.value
        self.value += 1.0
        return current

###############################################################################
class _StalledWorker:
    started_at = 0.0
    lifecycle_phase: str | None = None
    pid = 4321
    exitcode: int | None = None

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.terminated = False
        self.stop_called = False
        self.poll_count = 0

    # -------------------------------------------------------------------------
    def is_alive(self) -> bool:
        return not self.terminated

    # -------------------------------------------------------------------------
    def is_interrupted(self) -> bool:
        return self.stop_called

    # -------------------------------------------------------------------------
    def stop(self) -> None:
        self.stop_called = True

    # -------------------------------------------------------------------------
    def terminate(self) -> None:
        self.terminated = True

    # -------------------------------------------------------------------------
    def poll(self, timeout: float = 0.25) -> dict[str, object] | None:
        del timeout
        self.poll_count += 1
        if self.poll_count == 1:
            return {
                "type": "training_worker_lifecycle",
                "phase": "fit_entered",
                "status": "started",
                "pid": self.pid,
            }
        return None

    # -------------------------------------------------------------------------
    def join(self, timeout: float | None = None) -> None:
        del timeout

    # -------------------------------------------------------------------------
    def read_result(self, timeout: float = 0.5) -> None:
        del timeout
        return None

###############################################################################
def test_slow_first_batch_is_not_a_stall_before_its_deadline() -> None:
    watchdog = training_module._TrainingWatchdogState(started_at=0.0)
    watchdog.observe(
        {
            "type": "training_worker_lifecycle",
            "phase": "fit_entered",
            "status": "started",
        },
        1.0,
    )

    assert (
        watchdog.stall_reason(
            100.0,
            startup_timeout_seconds=10.0,
            phase_timeout_seconds=20.0,
            first_batch_timeout_seconds=120.0,
        )
        is None
    )

###############################################################################
def test_repeated_heartbeat_without_phase_or_batch_progress_stalls() -> None:
    watchdog = training_module._TrainingWatchdogState(started_at=0.0)
    first_event = {
        "type": "training_worker_lifecycle",
        "phase": "model_loading_started",
        "status": "started",
    }
    watchdog.observe(first_event, 1.0)
    watchdog.observe({**first_event, "status": "heartbeat"}, 2.0)
    watchdog.observe({**first_event, "status": "heartbeat"}, 3.0)

    assert (
        watchdog.stall_reason(
            4.0,
            startup_timeout_seconds=10.0,
            phase_timeout_seconds=2.0,
            first_batch_timeout_seconds=20.0,
        )
        == "worker_phase_timeout"
    )

###############################################################################
def test_monitor_terminates_stalled_worker_with_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _StalledWorker()
    manager = Mock()
    manager.should_stop.return_value = False
    monkeypatch.setattr(training_module, "get_job_manager", Mock(return_value=manager))

    with pytest.raises(JobExecutionError) as raised:
        training_module.monitor_training_process(
            "job-stalled",
            worker,  # type: ignore[arg-type]
            stop_timeout_seconds=0.0,
            startup_timeout_seconds=10.0,
            phase_timeout_seconds=2.0,
            first_batch_timeout_seconds=2.0,
            clock=_Clock(),
        )

    assert raised.value.code == "training_worker_stalled"
    assert raised.value.phase == "fit_entered"
    assert worker.stop_called is True
    assert worker.terminated is True
    assert any(
        call.args == ("job-stalled", {"worker_diagnostics": {
            "status": "stalled",
            "phase": "fit_entered",
            "pid": 4321,
            "exitcode": None,
            "reported": False,
            "reason": "first_batch_timeout",
            "phase_elapsed_seconds": 2.0,
            "elapsed_seconds": 3.0,
        }})
        for call in manager.update_result.call_args_list
    )










###############################################################################
class FakeProcessWorker:

    # -------------------------------------------------------------------------
    def __init__(
        self, *, interrupted: bool, max_alive_checks: int, exitcode: int | None
    ) -> None:
        self.interrupted = interrupted
        self.max_alive_checks = max_alive_checks
        self.alive_checks = 0
        self.terminated = False
        self.stop_called = False
        self.terminate_called = False
        self.join_called = False
        self.exitcode = exitcode

    # -------------------------------------------------------------------------
    def is_alive(self) -> bool:
        if self.terminated:
            return False
        self.alive_checks += 1
        return self.alive_checks <= self.max_alive_checks

    # -------------------------------------------------------------------------
    def is_interrupted(self) -> bool:
        return self.interrupted

    # -------------------------------------------------------------------------
    def stop(self) -> None:
        self.stop_called = True
        self.interrupted = True

    # -------------------------------------------------------------------------
    def terminate(self) -> None:
        self.terminate_called = True
        self.terminated = True

    # -------------------------------------------------------------------------
    def poll(self, timeout: float = 0.25):
        return None

    # -------------------------------------------------------------------------
    def join(self, timeout: float | None = None) -> None:
        self.join_called = True

    # -------------------------------------------------------------------------
    def read_result(self, timeout: float = 0.5):
        del timeout
        return None

###############################################################################
def test_training_cancel_requests_graceful_stop_before_forced_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = FakeProcessWorker(interrupted=False, max_alive_checks=1, exitcode=0)
    manager = Mock()
    manager.should_stop.return_value = True
    monkeypatch.setattr(training_module, "get_job_manager", Mock(return_value=manager))

    result = training_module.monitor_training_process(
        "job-cancelled",
        worker,
        stop_timeout_seconds=10.0,
    )

    assert worker.stop_called is True
    assert worker.terminate_called is False
    assert result == {}

###############################################################################
def test_training_cancel_forces_termination_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = FakeProcessWorker(interrupted=True, max_alive_checks=10, exitcode=1)
    manager = Mock()
    manager.should_stop.return_value = True
    monkeypatch.setattr(training_module, "get_job_manager", Mock(return_value=manager))

    result = training_module.monitor_training_process(
        "job-cancelled",
        worker,
        stop_timeout_seconds=0.0,
    )

    assert worker.terminate_called is True
    assert worker.join_called is True
    assert result == {}




















###############################################################################
def _child_reports_success(*, worker: WorkerChannels) -> None:
    worker.report_lifecycle("test_phase")
    worker.report_result({"result": {"completed": True}})

###############################################################################
def _raise_from_child(*, worker: WorkerChannels) -> None:
    del worker
    raise ValueError("child exploded")

###############################################################################
def _set_stop_event(*, worker: WorkerChannels) -> None:
    worker.stop_event.set()

###############################################################################
def test_child_failure_reports_lifecycle_and_traceback() -> None:
    progress_queue: queue.Queue[dict[str, object]] = queue.Queue()
    result_queue: queue.Queue[dict[str, object]] = queue.Queue()
    worker = WorkerChannels(progress_queue, result_queue, Event())

    with pytest.raises(ValueError, match="child exploded"):
        process_target(_raise_from_child, {}, worker)

    lifecycle = [
        progress_queue.get_nowait(),
        progress_queue.get_nowait(),
        progress_queue.get_nowait(),
    ]
    assert [message["status"] for message in lifecycle] == [
        "started",
        "started",
        "failed",
    ]
    assert [message["phase"] for message in lifecycle] == [
        "child_started",
        "target_started",
        "target_failed",
    ]
    assert lifecycle[2]["diagnostic"]["exception_type"] == "ValueError"  # type: ignore[index]

    failure = result_queue.get_nowait()
    assert failure["error"] == "child exploded"
    diagnostic = failure["worker_diagnostics"]
    assert diagnostic["status"] == "failed"  # type: ignore[index]
    assert diagnostic["message"] == "child exploded"  # type: ignore[index]
    assert "ValueError: child exploded" in diagnostic["traceback"]  # type: ignore[index]

###############################################################################
def test_child_cancellation_reports_cancelled_lifecycle() -> None:
    progress_queue: queue.Queue[dict[str, object]] = queue.Queue()
    result_queue: queue.Queue[dict[str, object]] = queue.Queue()
    worker = WorkerChannels(progress_queue, result_queue, Event())

    process_target(_set_stop_event, {}, worker)

    lifecycle = [
        progress_queue.get_nowait(),
        progress_queue.get_nowait(),
        progress_queue.get_nowait(),
    ]
    assert [message["status"] for message in lifecycle] == [
        "started",
        "started",
        "cancelled",
    ]
    assert [message["phase"] for message in lifecycle] == [
        "child_started",
        "target_started",
        "worker_cancelled",
    ]
    assert result_queue.empty()

###############################################################################
def test_plot_pressure_does_not_discard_critical_progress() -> None:
    progress_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=2)
    progress_queue.put({"type": "training_plot", "chart_point": {"batch": 1}})
    progress_queue.put({"type": "training_plot", "chart_point": {"batch": 2}})

    reporter = QueueProgressReporter(progress_queue)
    reporter({"type": "training_update", "progress_percent": 50})

    messages = [progress_queue.get_nowait(), progress_queue.get_nowait()]
    assert {message["type"] for message in messages} == {
        "training_plot",
        "training_update",
    }

###############################################################################
def test_lifecycle_message_includes_worker_correlation_and_timing() -> None:
    progress_queue: queue.Queue[dict[str, object]] = queue.Queue()
    result_queue: queue.Queue[dict[str, object]] = queue.Queue()
    worker = WorkerChannels(
        progress_queue,
        result_queue,
        Event(),
        job_id="job-correlated",
    )

    worker.report_lifecycle("model_loading_started")
    message = progress_queue.get_nowait()

    assert message["job_id"] == "job-correlated"
    assert message["pid"]
    assert message["phase"] == "model_loading_started"
    assert message["elapsed_seconds"] >= 0
    assert message["phase_elapsed_seconds"] == 0.0

###############################################################################
def test_training_callback_reports_fit_and_first_batch_boundaries() -> None:
    messages: list[dict[str, object]] = []
    callback = TrainingProgressCallback(
        messages.append,
        total_epochs=1,
        polling_interval=0.0,
    )
    callback.params = {"steps": 1}

    callback.on_train_begin()
    callback.on_train_batch_begin(0)
    callback.on_train_batch_end(0, {"loss": 1.0, "accuracy": 0.5})
    callback.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.1})

    lifecycle_phases = [
        message["phase"]
        for message in messages
        if message.get("type") == "training_worker_lifecycle"
    ]
    assert lifecycle_phases == [
        "fit_entered",
        "first_batch_entered",
        "first_batch_completed",
        "epoch_completed",
    ]
    assert any(message.get("type") == "training_update" for message in messages)

###############################################################################
def test_spawned_worker_delivers_lifecycle_and_result_without_queue_join() -> None:
    worker = ProcessWorker(progress_queue_size=1)
    messages: list[dict[str, object]] = []
    try:
        worker.start(
            target=_child_reports_success,
            kwargs={},
            job_id="job-spawned",
        )
        while worker.is_alive() or not messages:
            message = worker.poll(timeout=1.0)
            if message is not None:
                messages.append(message)
            if not worker.is_alive() and message is None:
                break
        worker.join(timeout=2.0)
        while True:
            message = worker.poll(timeout=0.0)
            if message is None:
                break
            messages.append(message)

        result = worker.read_result(timeout=1.0)
        assert result == {"result": {"completed": True}}
        assert any(message.get("phase") == "test_phase" for message in messages)
        assert all(
            message.get("job_id") == "job-spawned"
            for message in messages
            if message.get("type") == "training_worker_lifecycle"
        )
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2.0)
        worker.cleanup()

###############################################################################
class ExitedWorker:
    exitcode = 7
    pid = 1234

    # -------------------------------------------------------------------------
    def read_result(self, timeout: float = 0.5) -> None:
        del timeout
        return None

###############################################################################
def test_non_zero_worker_exit_is_typed_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = Mock()
    manager.should_stop.return_value = False
    monkeypatch.setattr(
        training_module,
        "get_job_manager",
        Mock(return_value=manager),
    )

    with pytest.raises(JobExecutionError) as raised:
        training_module.read_worker_result("job-exit", ExitedWorker())  # type: ignore[arg-type]

    assert str(raised.value) == "Training process exited with code 7"
    assert raised.value.code == "training_worker_exited"
    assert raised.value.phase == "worker_process"
    assert raised.value.recoverable is True
    manager.update_result.assert_called_once_with(
        "job-exit",
        {
            "worker_diagnostics": {
                "status": "exited",
                "phase": "worker_process",
                "pid": 1234,
                "exitcode": 7,
                "reported": False,
            }
        },
    )

###############################################################################
def test_reported_child_failure_keeps_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = Mock()
    manager.should_stop.return_value = False
    monkeypatch.setattr(
        training_module,
        "get_job_manager",
        Mock(return_value=manager),
    )

    ###############################################################################
    class FailedWorker:
        exitcode = 1
        pid = 5678

        # -------------------------------------------------------------------------
        def read_result(self, timeout: float = 0.5) -> dict[str, object]:
            del timeout
            return {
                "error": "child exploded",
                "worker_diagnostics": {
                    "status": "failed",
                    "pid": 5678,
                    "exception_type": "ValueError",
                    "message": "child exploded",
                    "traceback": "ValueError: child exploded",
                    "reported": True,
                },
            }

    with pytest.raises(JobExecutionError, match="child exploded") as raised:
        training_module.read_worker_result("job-failed", FailedWorker())  # type: ignore[arg-type]

    assert raised.value.code == "job_failed"
    manager.update_result.assert_called_once_with(
        "job-failed",
        {
            "worker_diagnostics": {
                "status": "failed",
                "phase": "worker_process",
                "pid": 5678,
                "exception_type": "ValueError",
                "reported": True,
                "exitcode": 1,
            }
        },
    )

###############################################################################
def test_normal_worker_result_is_returned_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = Mock()
    manager.should_stop.return_value = False
    monkeypatch.setattr(
        training_module,
        "get_job_manager",
        Mock(return_value=manager),
    )

    ###############################################################################
    class CompletedWorker:
        exitcode = 0
        pid = 4321

        # -------------------------------------------------------------------------
        def read_result(self, timeout: float = 0.5) -> dict[str, object]:
            del timeout
            return {"result": {"checkpoint_path": "checkpoint-a"}}

    result = training_module.read_worker_result("job-completed", CompletedWorker())  # type: ignore[arg-type]

    assert result == {"checkpoint_path": "checkpoint-a"}
    manager.update_result.assert_not_called()

###############################################################################
def test_missing_result_payload_is_a_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = Mock()
    manager.should_stop.return_value = False
    monkeypatch.setattr(
        training_module,
        "get_job_manager",
        Mock(return_value=manager),
    )

    ###############################################################################
    class MissingResultWorker:
        exitcode = 0
        pid = 1234

        # -------------------------------------------------------------------------
        def read_result(self, timeout: float = 0.5) -> None:
            del timeout
            return None

    with pytest.raises(JobExecutionError) as raised:
        training_module.read_worker_result("job-missing", MissingResultWorker())  # type: ignore[arg-type]

    assert raised.value.code == "training_worker_missing_result"

###############################################################################
def test_zero_exit_empty_result_is_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = Mock()
    manager.should_stop.return_value = False
    monkeypatch.setattr(
        training_module,
        "get_job_manager",
        Mock(return_value=manager),
    )

    ###############################################################################
    class EmptyResultWorker:
        exitcode = 0
        pid = 777

        # -------------------------------------------------------------------------
        def read_result(self, timeout: float = 0.5) -> dict[str, object]:
            del timeout
            return {"result": {}}

    with pytest.raises(JobExecutionError) as raised:
        training_module.read_worker_result("job-empty", EmptyResultWorker())  # type: ignore[arg-type]

    assert raised.value.code == "training_worker_missing_result"
    manager.update_result.assert_called_once_with(
        "job-empty",
        {
            "worker_diagnostics": {
                "status": "missing_result",
                "phase": "worker_process",
                "pid": 777,
                "exitcode": 0,
                "reported": True,
            }
        },
    )

###############################################################################
def test_xray_dataloaders_return_expected_tensor_contract(
    tmp_path: Path,
) -> None:
    image_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in image_paths:
        Image.new("RGB", (100, 100), "black").save(path)
    data = pd.DataFrame({"path": image_paths, "tokens": [[1, 2, 3], [4, 5, 6]]})
    loader = XRAYDataLoader(
        {
            "batch_size": 2,
            "inference_batch_size": 2,
            "use_img_augmentation": False,
            "dataloader_workers": 0,
        },
        shuffle=False,
    )

    (images, inputs), outputs = next(
        iter(loader.build_training_dataloader(data))
    )
    assert images.shape == (2, 224, 224, 3)
    assert inputs.shape == (2, 2)
    assert outputs.shape == (2, 2)
    assert images.dtype == torch.float32
    assert inputs.dtype == torch.int64
    assert outputs.dtype == torch.int64

    inference_images = next(iter(loader.build_inference_dataloader(data)))
    assert inference_images.shape == (2, 224, 224, 3)
    assert inference_images.dtype == torch.float32

###############################################################################
def test_xray_training_loader_runs_with_spawned_workers(tmp_path: Path) -> None:
    # Windows DataLoader workers are spawned and must unpickle the transforms.
    image_paths = [tmp_path / f"image_{index}.png" for index in range(4)]
    for path in image_paths:
        Image.new("L", (64, 48), "white").save(path)
    data = pd.DataFrame({"path": image_paths, "tokens": [[1, 2, 3]] * 4})
    loader = XRAYDataLoader(
        {"batch_size": 2, "dataloader_workers": 2, "prefetch_factor": 1},
        shuffle=False,
    ).build_training_dataloader(data)

    pickle.dumps(loader.dataset)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0][0][0].shape == (2, 224, 224, 3)
