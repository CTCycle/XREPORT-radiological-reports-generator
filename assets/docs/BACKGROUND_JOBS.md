# Background Job Management

XREPORT uses a centralized, thread-based job manager to run long operations without blocking the FastAPI request thread. It powers dataset processing, training, inference, dataset validation, and checkpoint evaluation.

Core implementation:
- `XREPORT/server/services/jobs.py` (`JobManager`, global `job_manager`)
- `XREPORT/server/domain/jobs.py` (`JobState`, job response models)

## Core Concepts

### Threading model
- Each job runs in a daemon `threading.Thread`.
- Jobs are cooperative-cancellation: threads are not force-killed by default.
- Training is a special case: the job thread supervises a separate `ProcessWorker` child process (`XREPORT/server/learning/training/worker.py`) for heavy ML execution.

### Job state
Each job is tracked in a `JobState` object with:
- `job_id` (8-char UUID)
- `job_type`
- `status` (`pending`, `running`, `completed`, `failed`, `cancelled`)
- `progress` (0.0 to 100.0)
- `result` (merged partial/final payload)
- `error`
- internal cancellation flag (`stop_requested`)

## Usage Pattern

### 1. Import the singleton
```python
from XREPORT.server.services.jobs import job_manager
```

### 2. Define a blocking runner
Prefer runners that accept `job_id`; `JobManager.start_job` injects it automatically when supported.

```python
def run_my_job(payload: dict, job_id: str) -> dict:
    if job_manager.should_stop(job_id):
        return {}

    # Optional live updates for polling clients
    job_manager.update_progress(job_id, 50.0)
    job_manager.update_result(job_id, {"stage": "halfway"})

    result = do_blocking_work(payload)
    return {"result": result}
```

### 3. Start from endpoint code
```python
if job_manager.is_job_running("my_job_type"):
    raise HTTPException(status_code=409, detail="Job already running")

job_id = job_manager.start_job(
    job_type="my_job_type",
    runner=run_my_job,
    kwargs={"payload": request.model_dump()},
)
```

### 4. Expose standard polling/cancel endpoints
- `GET .../jobs/{job_id}` -> `job_manager.get_job_status(job_id)`
- `DELETE .../jobs/{job_id}` -> `job_manager.cancel_job(job_id)`

## Result Merging Behavior

XREPORT supports partial result updates while a job is running:
- `update_result(job_id, patch)` merges into the existing result dict.
- when the runner returns, returned payload is merged with partial state.
- this enables live UI progress/metrics without losing final summary values.

## Cancellation Semantics

- `cancel_job` on a running job sets `stop_requested=True`.
- runner code must check `should_stop(job_id)` and exit cleanly.
- for training jobs, route logic also triggers `ProcessWorker.stop()` and can escalate to `terminate()` after timeout in `monitor_training_process`.

## Frontend Interaction

All long-running workflows use polling:
1. Start endpoint returns `job_id` (+ optional `poll_interval`).
2. UI polls corresponding `.../jobs/{job_id}` endpoint.
3. UI stops polling on terminal status (`completed`, `failed`, `cancelled`).
