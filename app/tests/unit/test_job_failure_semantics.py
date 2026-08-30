from __future__ import annotations

from server.services.inference import map_inference_failure
from server.services.jobs import JobExecutionError, JobManager


###############################################################################
def _snapshot_failure() -> dict[str, object]:
    raise RuntimeError("checkpoint snapshot could not be loaded")


###############################################################################
def test_generic_job_failure_uses_generic_execution_taxonomy() -> None:
    manager = JobManager()
    job_id = manager.start_job(job_type="validation", runner=_snapshot_failure)
    thread = manager.threads[job_id]
    thread.join(timeout=1.0)

    status = manager.get_job_status(job_id)
    assert status is not None
    assert status["status"] == "failed"
    assert status["result"]["failure"] == {
        "code": "job_failed",
        "message": "checkpoint snapshot could not be loaded",
        "phase": "execution",
        "recoverable": True,
    }


###############################################################################
def test_inference_failure_taxonomy_is_injected_by_feature() -> None:
    manager = JobManager()
    job_id = manager.start_job(
        job_type="inference",
        runner=_snapshot_failure,
        failure_mapper=map_inference_failure,
    )
    thread = manager.threads[job_id]
    thread.join(timeout=1.0)

    status = manager.get_job_status(job_id)
    assert status is not None
    assert status["status"] == "failed"
    assert status["result"]["failure"] == {
        "code": "model_load_failed",
        "message": "checkpoint snapshot could not be loaded",
        "phase": "loading",
        "recoverable": True,
    }


###############################################################################
def test_typed_job_failure_is_used_without_a_mapper() -> None:
    manager = JobManager()

    def runner() -> dict[str, object]:
        raise JobExecutionError(
            "validation artifact is unavailable",
            code="artifact_missing",
            phase="validation",
            recoverable=False,
        )

    job_id = manager.start_job(job_type="validation", runner=runner)
    thread = manager.threads[job_id]
    thread.join(timeout=1.0)

    status = manager.get_job_status(job_id)
    assert status is not None
    assert status["result"]["failure"] == {
        "code": "artifact_missing",
        "message": "validation artifact is unavailable",
        "phase": "validation",
        "recoverable": False,
    }
