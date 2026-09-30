from __future__ import annotations

import os
import queue
from threading import Event
from unittest.mock import Mock

import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

from server.models.callbacks import TrainingProgressCallback
from server.services import training as training_module
from server.services.jobs import JobExecutionError
from server.services.training_worker import (
    ProcessWorker,
    QueueProgressReporter,
    WorkerChannels,
    process_target,
)


def _child_reports_success(*, worker: WorkerChannels) -> None:
    worker.report_lifecycle("test_phase")
    worker.report_result({"result": {"completed": True}})


def _raise_from_child(*, worker: WorkerChannels) -> None:
    del worker
    raise ValueError("child exploded")


def _set_stop_event(*, worker: WorkerChannels) -> None:
    worker.stop_event.set()


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


class ExitedWorker:
    exitcode = 7
    pid = 1234

    def read_result(self, timeout: float = 0.5) -> None:
        del timeout
        return None


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

    class FailedWorker:
        exitcode = 1
        pid = 5678

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

    class CompletedWorker:
        exitcode = 0
        pid = 4321

        def read_result(self, timeout: float = 0.5) -> dict[str, object]:
            del timeout
            return {"result": {"checkpoint_path": "checkpoint-a"}}

    result = training_module.read_worker_result("job-completed", CompletedWorker())  # type: ignore[arg-type]

    assert result == {"checkpoint_path": "checkpoint-a"}
    manager.update_result.assert_not_called()


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

    class MissingResultWorker:
        exitcode = 0
        pid = 1234

        def read_result(self, timeout: float = 0.5) -> None:
            del timeout
            return None

    with pytest.raises(JobExecutionError) as raised:
        training_module.read_worker_result("job-missing", MissingResultWorker())  # type: ignore[arg-type]

    assert raised.value.code == "training_worker_missing_result"


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

    class EmptyResultWorker:
        exitcode = 0
        pid = 777

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
