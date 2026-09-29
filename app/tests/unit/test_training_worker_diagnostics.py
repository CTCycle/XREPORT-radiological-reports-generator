from __future__ import annotations

import queue
from threading import Event
from unittest.mock import Mock

import pytest

from server.services import training as training_module
from server.services.jobs import JobExecutionError
from server.services.training_worker import WorkerChannels, process_target


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


class ExitedWorker:
    exitcode = 7
    pid = 1234

    def read_result(self) -> None:
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

        def read_result(self) -> dict[str, object]:
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

        def read_result(self) -> dict[str, object]:
            return {"result": {"checkpoint_path": "checkpoint-a"}}

    result = training_module.read_worker_result("job-completed", CompletedWorker())  # type: ignore[arg-type]

    assert result == {"checkpoint_path": "checkpoint-a"}
    manager.update_result.assert_not_called()
