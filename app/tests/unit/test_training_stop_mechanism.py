from __future__ import annotations

import os
from unittest.mock import Mock

import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

from server.services import training as training_module


###############################################################################
class FakeProcessWorker:
    def __init__(self, *, interrupted: bool, max_alive_checks: int, exitcode: int | None) -> None:
        self.interrupted = interrupted
        self.max_alive_checks = max_alive_checks
        self.alive_checks = 0
        self.terminated = False
        self.stop_called = False
        self.terminate_called = False
        self.join_called = False
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        if self.terminated:
            return False
        self.alive_checks += 1
        return self.alive_checks <= self.max_alive_checks

    def is_interrupted(self) -> bool:
        return self.interrupted

    def stop(self) -> None:
        self.stop_called = True
        self.interrupted = True

    def terminate(self) -> None:
        self.terminate_called = True
        self.terminated = True

    def poll(self, timeout: float = 0.25):
        return None

    def join(self, timeout: float | None = None) -> None:
        self.join_called = True

    def read_result(self):
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
