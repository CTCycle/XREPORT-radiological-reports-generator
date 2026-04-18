from __future__ import annotations

import os
from unittest.mock import Mock

import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

from XREPORT.server.learning.callbacks import TrainingInterruptCallback, WorkerInterrupted
from XREPORT.server.api import training as training_routes


###############################################################################
class FakeProcessWorker:
    def __init__(
        self,
        interrupted: bool,
        max_alive_checks: int = 5,
        exitcode: int | None = 0,
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
    def read_result(self):
        return None


###############################################################################
class BrokenWorker:
    # -------------------------------------------------------------------------
    def is_interrupted(self) -> bool:
        raise RuntimeError("Worker channel unavailable")


###############################################################################
def test_monitor_starts_timeout_even_if_worker_already_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = FakeProcessWorker(interrupted=True, max_alive_checks=10, exitcode=1)
    job_manager_mock = Mock()
    job_manager_mock.should_stop.return_value = True
    monkeypatch.setattr(
        training_routes,
        "get_job_manager",
        Mock(return_value=job_manager_mock),
    )

    result = training_routes.monitor_training_process(
        "job-cancelled",
        worker,
        stop_timeout_seconds=0.0,
    )

    assert worker.terminate_called is True
    assert worker.join_called is True
    assert result == {}


# -------------------------------------------------------------------------
def test_monitor_requests_graceful_stop_before_forcing_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = FakeProcessWorker(interrupted=False, max_alive_checks=1, exitcode=0)
    job_manager_mock = Mock()
    job_manager_mock.should_stop.return_value = True
    monkeypatch.setattr(
        training_routes,
        "get_job_manager",
        Mock(return_value=job_manager_mock),
    )

    result = training_routes.monitor_training_process(
        "job-cancelled",
        worker,
        stop_timeout_seconds=10.0,
    )

    assert worker.stop_called is True
    assert result == {}


# -------------------------------------------------------------------------
def test_interrupt_callback_raises_worker_interrupted() -> None:
    callback = TrainingInterruptCallback(worker=FakeProcessWorker(interrupted=True))

    with pytest.raises(WorkerInterrupted):
        callback.raise_if_interrupted()


# -------------------------------------------------------------------------
def test_interrupt_callback_ignores_worker_probe_failures() -> None:
    callback = TrainingInterruptCallback(worker=BrokenWorker())
    callback.raise_if_interrupted()
