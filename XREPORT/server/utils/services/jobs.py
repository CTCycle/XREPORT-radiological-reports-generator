"""Background job manager for long-running operations."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from collections.abc import Callable

from XREPORT.server.utils.logger import logger


@dataclass
class JobState:
    job_id: str
    job_type: str
    status: str
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=monotonic)
    completed_at: float | None = None
    stop_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # -------------------------------------------------------------------------
    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    # -------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "job_type": self.job_type,
                "status": self.status,
                "progress": self.progress,
                "result": self.result,
                "error": self.error,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
            }


###############################################################################
class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, JobState] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

    # -------------------------------------------------------------------------
    def start_job(
        self,
        job_type: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())[:8]
        state = JobState(job_id=job_id, job_type=job_type, status="pending")

        with self.lock:
            self.jobs[job_id] = state

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, runner, args, kwargs or {}),
            daemon=True,
        )

        with self.lock:
            self.threads[job_id] = thread

        state.update(status="running")
        thread.start()

        logger.info("Started job %s (type=%s)", job_id, job_type)
        return job_id

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return None
        return state.snapshot()

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> bool:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return False
        if state.status not in ("pending", "running"):
            return False
        state.update(stop_requested=True, status="cancelled", completed_at=monotonic())
        logger.info("Cancelled job %s", job_id)
        return True

    # -------------------------------------------------------------------------
    def is_job_running(self, job_type: str | None = None) -> bool:
        with self.lock:
            for state in self.jobs.values():
                if state.status in ("pending", "running"):
                    if job_type is None or state.job_type == job_type:
                        return True
        return False

    # -------------------------------------------------------------------------
    def list_jobs(self, job_type: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            states = list(self.jobs.values())
        results = []
        for state in states:
            if job_type is None or state.job_type == job_type:
                results.append(state.snapshot())
        return results

    # -------------------------------------------------------------------------
    def should_stop(self, job_id: str) -> bool:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return True
        return state.stop_requested

    # -------------------------------------------------------------------------
    def update_progress(self, job_id: str, progress: float) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state:
            state.update(progress=min(100.0, max(0.0, progress)))

    # -------------------------------------------------------------------------
    def _run_job(
        self,
        job_id: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return

        try:
            result = runner(*args, **kwargs)
            if state.stop_requested:
                state.update(status="cancelled", completed_at=monotonic())
            else:
                state.update(
                    status="completed",
                    result=result,
                    progress=100.0,
                    completed_at=monotonic(),
                )
                logger.info("Job %s completed successfully", job_id)
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc).split("\n")[0][:200]
            state.update(status="failed", error=error_msg, completed_at=monotonic())
            logger.error("Job %s failed: %s", job_id, error_msg)
            logger.debug("Job %s error details", job_id, exc_info=True)


###############################################################################
job_manager = JobManager()
