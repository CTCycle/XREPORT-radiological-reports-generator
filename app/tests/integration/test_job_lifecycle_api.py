from __future__ import annotations

from collections.abc import Iterator
from threading import Event

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from server.api.errors import register_service_error_handlers
from server.api.jobs import JobsEndpoint
from server.services.jobs import JobExecutionError, JobManager

###############################################################################
@pytest.fixture
def job_api() -> Iterator[tuple[TestClient, JobManager]]:
    manager = JobManager()
    router = APIRouter(prefix="/jobs")
    JobsEndpoint(router=router, job_manager=manager).add_routes()

    application = FastAPI()
    register_service_error_handlers(application)
    application.include_router(router, prefix="/api")

    with TestClient(application) as client:
        yield client, manager


###############################################################################
def _join_job(manager: JobManager, job_id: str) -> None:
    thread = manager.threads[job_id]
    thread.join(timeout=5.0)
    assert not thread.is_alive(), f"job {job_id} did not finish within five seconds"


###############################################################################
def test_job_api_lists_filters_and_polls_to_completion(
    job_api: tuple[TestClient, JobManager],
) -> None:
    client, manager = job_api
    runner_entered = Event()
    release_runner = Event()

    def controlled_runner() -> dict[str, object]:
        runner_entered.set()
        if not release_runner.wait(timeout=5.0):
            raise TimeoutError("test runner was not released")
        return {"receipt": "completed"}

    validation_job_id = manager.start_job(
        job_type="validation",
        runner=controlled_runner,
        poll_interval=0.25,
    )
    assert runner_entered.wait(timeout=2.0)

    training_job_id = manager.start_job(
        job_type="training",
        runner=lambda: {"receipt": "training-completed"},
        poll_interval=0.25,
    )
    _join_job(manager, training_job_id)

    try:
        all_jobs = client.get("/api/jobs")
        assert all_jobs.status_code == 200
        assert {job["job_id"] for job in all_jobs.json()["jobs"]} == {
            validation_job_id,
            training_job_id,
        }

        by_type = client.get("/api/jobs", params={"job_type": "validation"})
        assert by_type.status_code == 200
        assert [job["job_id"] for job in by_type.json()["jobs"]] == [
            validation_job_id
        ]

        by_status = client.get("/api/jobs", params={"status": "running"})
        assert by_status.status_code == 200
        assert [job["job_id"] for job in by_status.json()["jobs"]] == [
            validation_job_id
        ]

        running = client.get(f"/api/jobs/{validation_job_id}")
        assert running.status_code == 200
        assert running.json()["status"] == "running"

        completed_jobs = client.get("/api/jobs", params={"status": "completed"})
        assert completed_jobs.status_code == 200
        assert [job["job_id"] for job in completed_jobs.json()["jobs"]] == [
            training_job_id
        ]
    finally:
        release_runner.set()

    _join_job(manager, validation_job_id)
    completed = client.get(f"/api/jobs/{validation_job_id}")
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["progress"] == 100.0
    assert completed.json()["result"] == {"receipt": "completed"}

    cannot_cancel_completed = client.delete(f"/api/jobs/{validation_job_id}")
    assert cannot_cancel_completed.status_code == 200
    assert cannot_cancel_completed.json() == {
        "job_id": validation_job_id,
        "success": False,
        "message": "Job cannot be cancelled",
    }


###############################################################################
def test_job_api_returns_not_found_for_unknown_job(
    job_api: tuple[TestClient, JobManager],
) -> None:
    client, _manager = job_api

    status_response = client.get("/api/jobs/missing-job")
    assert status_response.status_code == 404
    assert status_response.json() == {"detail": "Job not found: missing-job"}

    cancel_response = client.delete("/api/jobs/missing-job")
    assert cancel_response.status_code == 404
    assert cancel_response.json() == {"detail": "Job not found: missing-job"}


###############################################################################
def test_job_api_cancellation_stays_active_until_runner_exits(
    job_api: tuple[TestClient, JobManager],
) -> None:
    client, manager = job_api
    runner_entered = Event()
    release_runner = Event()

    def controlled_runner() -> dict[str, object]:
        runner_entered.set()
        if not release_runner.wait(timeout=5.0):
            raise TimeoutError("test runner was not released")
        return {}

    job_id = manager.start_job(
        job_type="training",
        runner=controlled_runner,
        poll_interval=0.25,
    )
    assert runner_entered.wait(timeout=2.0)

    try:
        cancellation = client.delete(f"/api/jobs/{job_id}")
        assert cancellation.status_code == 200
        assert cancellation.json() == {
            "job_id": job_id,
            "success": True,
            "message": "Cancellation requested",
        }

        still_running = client.get(f"/api/jobs/{job_id}")
        assert still_running.status_code == 200
        assert still_running.json()["status"] == "running"
    finally:
        release_runner.set()

    _join_job(manager, job_id)
    cancelled = client.get(f"/api/jobs/{job_id}")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


###############################################################################
def test_job_api_preserves_typed_recoverable_failure_timeline(
    job_api: tuple[TestClient, JobManager],
) -> None:
    client, manager = job_api

    def failing_runner() -> dict[str, object]:
        raise JobExecutionError(
            "validation artifact is temporarily unavailable",
            code="artifact_unavailable",
            phase="validation",
            recoverable=True,
        )

    job_id = manager.start_job(
        job_type="validation",
        runner=failing_runner,
        poll_interval=0.25,
    )
    _join_job(manager, job_id)

    failed = client.get(f"/api/jobs/{job_id}")
    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    assert failed.json()["error"] == "validation artifact is temporarily unavailable"
    assert failed.json()["result"]["failure"] == {
        "code": "artifact_unavailable",
        "message": "validation artifact is temporarily unavailable",
        "phase": "validation",
        "recoverable": True,
    }

    failed_jobs = client.get("/api/jobs", params={"status": "failed"})
    assert failed_jobs.status_code == 200
    assert [job["job_id"] for job in failed_jobs.json()["jobs"]] == [job_id]
