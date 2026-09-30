from __future__ import annotations

from unittest.mock import Mock

import pytest

from server.services import training as training_module
from server.services.jobs import JobExecutionError


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 1.0
        return current


class _StalledWorker:
    started_at = 0.0
    lifecycle_phase: str | None = None
    pid = 4321
    exitcode: int | None = None

    def __init__(self) -> None:
        self.terminated = False
        self.stop_called = False
        self.poll_count = 0

    def is_alive(self) -> bool:
        return not self.terminated

    def is_interrupted(self) -> bool:
        return self.stop_called

    def stop(self) -> None:
        self.stop_called = True

    def terminate(self) -> None:
        self.terminated = True

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

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def read_result(self, timeout: float = 0.5) -> None:
        del timeout
        return None


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
