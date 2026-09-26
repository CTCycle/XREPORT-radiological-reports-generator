# S51 current contention regression — 2026-09-26

Last updated: 2026-09-26

Source revision: 69737f8f0acec5d225f884cea70e7bb26b2d8664

Command:

    app/server/.venv/Scripts/python.exe -m pytest -c app/server/pyproject.toml app/tests/unit/test_job_start_concurrency.py app/tests/unit/test_job_cancellation_semantics.py app/tests/unit/test_job_failure_semantics.py app/tests/unit/test_training_stop_mechanism.py app/tests/unit/test_validation_job_semantics.py app/tests/unit/test_preparation_image_scanning.py -q --basetemp runtimes/validation-s50-s53-20260926/pytest-tmp/s51 -o cache_dir=runtimes/validation-s50-s53-20260926/pytest-cache

Result: 25 passed in 1.98s.

The concurrent-start test constructed the real TrainingService and JobManager,
synchronized two requests after their preflight checks, and observed exactly
one accepted training job and one ConflictError. The accepted runner completed
after release, its thread joined, and the manager reported no active training.
Adjacent cancellation, failure, validation, and preparation checks also passed.

This revalidates the current atomic require-idle admission fix. It does not
promote S51 beyond PARTIAL: the other feature-service overlap modes, UI
close/reopen behavior, real CUDA/process contention, resource high-water
measurement, and no-stuck-job coverage remain outside this run.
