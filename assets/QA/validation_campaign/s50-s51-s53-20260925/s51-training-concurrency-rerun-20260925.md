# S51 training concurrent-start regression rerun

Date: 2026-09-25 (local time)
Scope: focused same-type concurrent training-start contention only.

## Command and isolation

Ran `app/server/.venv/Scripts/python.exe -m pytest -c app/server/pyproject.toml app/tests/unit/test_job_start_concurrency.py -q` with `XREPORT_RESOURCES_DIR` set to `runtimes/cache/release-validation-20260925/resilience-rerun-20260925-01/resources`, pytest basetemp at `.../pytest-tmp`, and pytest cache at `.../pytest-cache`. The sibling `runtimes/cache/release-validation-20260925/resources` already existed and was left untouched. No service or port was launched.

## Result

Exit code 0; `1 passed in 0.86s`.

The test constructs the actual `TrainingService` and `JobManager`, synchronizes both concurrent requests after their preflight running-state reads, and blocks the single accepted runner. It asserts exactly one `JobStartResponse` and one `ConflictError` with detail `Training is already in progress`, exactly one running training job, then releases the runner and asserts the manager thread joins and is no longer alive, the job is completed, and training is no longer active. The two-worker `ThreadPoolExecutor` exits its context normally.

Post-run process check found no Python process whose command line matched `test_job_start_concurrency.py`. No browser or external endpoint was used.

## Coverage boundary

This directly validates the S51 same-type training double-start regression in the current working tree. The focused harness imports and constructs only `TrainingService` and the training module; it has no evaluation/processing parametrization or fixtures. No analogous validation/checkpoint-evaluation or processing duplicate-start run was performed from this harness.
