from __future__ import annotations

import threading
import time

from XREPORT.server.domain.jobs import JobState
from XREPORT.server.services.jobs import JobManager


###############################################################################
def blocking_runner(release: threading.Event) -> dict[str, object]:
    release.wait(timeout=3.0)
    return {}


###############################################################################
def test_cancel_running_job_marks_stop_requested_only() -> None:
    manager = JobManager()
    release = threading.Event()

    job_id = manager.start_job(
        job_type="training",
        runner=blocking_runner,
        kwargs={"release": release},
    )

    assert manager.cancel_job(job_id) is True
    status = manager.get_job_status(job_id)
    assert status is not None
    assert status["status"] in {"pending", "running"}

    with manager.lock:
        internal_state = manager.jobs[job_id]
        assert internal_state.stop_requested is True
    release.set()
    thread = manager.threads.get(job_id)
    if thread is not None:
        thread.join(timeout=1.0)
    time.sleep(0.01)

    final_status = manager.get_job_status(job_id)
    assert final_status is not None
    assert final_status["status"] == "cancelled"


# -----------------------------------------------------------------------------
def test_cancel_pending_job_transitions_to_cancelled() -> None:
    manager = JobManager()
    job_id = "manual_pending"
    with manager.lock:
        manager.jobs[job_id] = JobState(
            job_id=job_id,
            job_type="training",
            status="pending",
        )

    assert manager.cancel_job(job_id) is True
    status = manager.get_job_status(job_id)
    assert status is not None
    assert status["status"] == "cancelled"
