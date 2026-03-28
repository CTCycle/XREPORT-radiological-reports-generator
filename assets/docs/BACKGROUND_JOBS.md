# Background Job Management

XREPORT uses a centralized, thread-based job manager for long-running operations.

Primary implementation:
- `XREPORT/server/services/jobs.py` (`JobManager`, global `job_manager`)
- `XREPORT/server/domain/jobs.py` (job state and response models)

## 1. Job Types

Current long-running job types include:
- `preparation`
- `training`
- `inference`
- `validation`
- `checkpoint_evaluation`

Routes enforce single-running-job semantics per job type where needed.

## 2. Threading and Lifecycle

### 2.1 Execution model
- Each job runs in a daemon `threading.Thread`.
- Cancellation is cooperative by default (`stop_requested` flag).
- Training is a special case: the job thread supervises a `ProcessWorker` subprocess for heavy compute.

### 2.2 State model
Each job stores:
- `job_id` (short UUID)
- `job_type`
- `status`: `pending`, `running`, `completed`, `failed`, `cancelled`
- `progress` (`0.0` to `100.0`)
- `result` (dict payload, supports incremental merge)
- `error` (terminal failure message)

## 3. API Usage Pattern

### 3.1 Start
A start endpoint calls:
```python
job_id = job_manager.start_job(job_type="...", runner=..., kwargs={...})
```

### 3.2 Poll
Status endpoint returns:
```python
job_manager.get_job_status(job_id)
```

### 3.3 Cancel
Cancel endpoint calls:
```python
job_manager.cancel_job(job_id)
```

## 4. Live Result Merging

During execution, runners can send partial updates:
- `job_manager.update_progress(job_id, value)`
- `job_manager.update_result(job_id, patch_dict)`

Partial updates merge into the final `result`, so UIs can display live progress without losing final payload fields.

## 5. Cancellation Rules

- `cancel_job` marks the job with `stop_requested=True`.
- Runner logic must check `job_manager.should_stop(job_id)` and exit cleanly.
- Training cancellation also calls `ProcessWorker.stop()` and may escalate to termination if needed.

## 6. Frontend Contract

All long tasks follow the same contract:
1. Start endpoint returns `job_id` and may return `poll_interval`.
2. Frontend polls `GET /.../jobs/{job_id}`.
3. Polling stops when status is terminal.

Terminal statuses:
- `completed`
- `failed`
- `cancelled`

