# S50, S51, and S53 Resilience and Baseline Validation

Date: 2026-09-25

## Revision and environment

- Code baseline: `7d8eb47165258bc0d994b3946d75a69940fbf816` on `develop`; the checkout was clean at the start of this slice.
- Platform: Windows 11 Pro build 26200, AMD Ryzen 5 5600H (6 cores / 12 logical processors), 33,667,739,648 bytes physical RAM, NVIDIA GeForce RTX 3060 Laptop GPU. Prior S31 evidence records 6 GiB VRAM for this device.
- Python: the existing `app/server/.venv` interpreter.
- No app service, listener, browser, or packaged executable was started. Database, model metadata, and partial snapshot probes used disposable roots under `runtimes/cache/pytest-tmp/s50-s51-s53-20260925`; those state roots were removed. No canonical database or model snapshot was changed.
- The primary task later made concurrent source edits while this summary was being prepared. The findings below apply to the baseline above and do not validate those later edits.

## S50 — Restart and corruption recovery

Result: **PARTIAL** for current-code startup and fail-closed corruption behavior.

The persisted-state boundary reviewed in the project documentation is:

| State | Location under the configured resources root | Expected behavior |
| --- | --- | --- |
| SQLite database and application records | `database.db` | Persistent; startup applies/checks Alembic migrations and rejects incompatible state. |
| Checkpoint artifacts and registry | `checkpoints/` plus database registry | Persistent; registered artifact identity is database-owned. |
| Public model installation and lifecycle metadata | `models/huggingface/installed/`, `metadata/`, `staging/`, and `rollback/` | Installed snapshots persist; partial/corrupt states must not be promoted as ready. |
| Tokenizers, templates, and logs | `tokenizers/`, `templates/`, and `logs/` | Persistent resources/logs. |
| Hub, Torch, Keras, and other runtime caches | Separate configured cache root | Transient; not the authoritative installed model or application state. |
| Active generic jobs and uploaded inference image bytes | Process memory | Not durable across backend-process restart; completed workflow records have separate database persistence. |

The durable-path rules are documented in [persistence](../../../docs/architecture/persistence.md), [runtime startup](../../../docs/runtime/startup.md), and [execution and data flow](../../../docs/architecture/execution_and_data_flow.md).

Current-revision checks:

- Focused pytest command covered test_database_initialization.py, test_startup_validation.py, test_model_installation.py, test_preparation_image_scanning.py, and test_validation_job_semantics.py along with the S51 lifecycle files: **46 passed in 4.60s**.
- Fresh disposable SQLite initialization reached Alembic head `e91a4f6c2d73`; repeated initialization was idempotent and `PRAGMA integrity_check` returned `ok`.
- A disposable file containing invalid SQLite bytes failed startup twice with `Database startup migration failed: (sqlite3.DatabaseError) file is not a database`. Its SHA-256 stayed `f1e05a024158ae026838d930cc306fb0dafb034145a151234a9a43fdb3945989` across both failures; the file was not reset or stamped.
- Deleted-image checks passed: `test_validate_img_paths_fails_closed_for_deleted_images` and `test_validation_job_fails_when_no_image_paths_remain` returned typed integrity/input failures.
- A disposable active model snapshot containing its config and tokenizer but missing its required weight returned no active target; metadata became `state=corrupt` / `integrity=failed`, and the partial snapshot was preserved.
- Malformed JSON model metadata raised `InstallationError` and its bytes were unchanged. Existing installer regressions also passed for a hash-corrupt active snapshot, interrupted staging, and resumable first install.
- Cold and repeat SQLite initializer times were 0.125320s and 0.039890s on the current checkout. These are database-initializer measurements, not app-process startup times.

Earlier end-to-end restart evidence exists but was captured on earlier source revisions: [S22](../s22/summary-20260923.md) restarted the official source backend and retained synthetic processing metadata and SQLite integrity; [S31](../s31/summary-20260924.md) restarted the official launcher and reused the exact pinned model snapshot. S50 remains partial because this slice did not restart the current full application with user-visible persisted records, exercise recovery/restore after the injected corruption, or inspect the resulting UI/API error presentation.

## S51 — Concurrent operations and resource contention

Result: **FAIL** for atomic same-type job admission; cancellation and sequential overlap behavior passed in synthetic service runs.

- The focused 46-test run passed test_job_cancellation_semantics.py, test_job_failure_semantics.py, and test_training_stop_mechanism.py.
- Deterministic in-process harness used the real TrainingService.start_training and JobManager, a fake two-row repository, and a blocking synthetic runner. A barrier ensured two simultaneous requests both observed no active training before either started a job. Both were accepted and simultaneously reported `running` (`426dbc60` and `f83ec910`), then both completed when released. A sequential second request while a first job was held did raise ConflictError.
- Rapid cancellation/restart was exercised with the real service/job manager and synthetic runner: cancellation was accepted; an immediate retry was rejected while the first cooperative runner was still active; after it reached `cancelled`, a new job was accepted and completed (`c7a6462f` then `5e0fd7c1`). All synthetic job threads stopped.
- The source inspection found the same check-then-start structure in training/resume, dataset processing, validation, and checkpoint evaluation. Only training admission was deterministically reproduced here; simultaneous starts for the other services remain unmeasured.
- No rendered UI close/reopen during an active operation, real CUDA contention, process snapshot, or working-set / CUDA high-water measurement was performed in this slice. Prior S25 evidence in [S23–S26](../s23-s26/summary-20260923.md) observed the real CUDA worker exit after cancellation and then completed a resume; it did not exercise simultaneous training admission.

The reproduced training double-start violates the intended single-active-training guard and risks overlapping GPU work. The primary task should retest its pending concurrency fix against both the service race and the new regression test before S51 can pass.

## S53 — Performance and long-operation baseline

Result: **PARTIAL**. Current evidence is a tiny SQLite startup sample plus historical technical-path timings, not a complete release baseline.

Current hardware identity was read from Windows CIM: Ryzen 5 5600H, 6 cores / 12 logical processors, 32 GiB installed RAM, RTX 3060 Laptop GPU. The previous S31 validation identified the same GPU as 6 GiB VRAM.

Historical samples are useful for context only:

- [S31](../s31/summary-20260924.md), code baseline `80e8d7aea3530c31d7557445685591533a27b3c5`, Windows 11 build 26200, Python 3.14.7, PyTorch 2.10.0+cu130, exact CXRMate Multi revision `330721b9aa5bba201a3eb88eba4dd9a6607f3e7a`: first install 60.36s, first one-image generation 7.32s, post-restart reuse generation 9.01s, deletion 0.8s. The run used an isolated install and one public image.
- [S24/S25 receipt](../s23-s26/s24-s25-training.json), synthetic eight-row CUDA fixture, source baseline `3846609cc68646b501cd480c2a11938fa2278f8b`: the one-epoch session ran from log time 19:53:38 to 19:55:50 (132s); the whole job ran from 19:52:16 to 19:55:54 (218s). An unrelated llama-server also used the GPU during part of this run, so GPU utilization/memory were not attributable to XREPORT alone.
- S22 processing covered eight rows and restart persistence but did not instrument wall time or peak memory; see [S22 limits](../s22/summary-20260923.md).

The prior S31 isolated installation root and the exact model snapshot in the shared runtime cache are absent in this checkout. The canonical resource contains a known one-file manifest mismatch recorded in [S33/S52](../s33-s52-20260924/summary.md) as ISSUE-006, so it was not used for fresh performance measurements. This slice did not measure current full app readiness, dataset processing, model loading, inference generation, process RAM/CUDA peaks, or packaged startup. S53 cannot be promoted beyond PARTIAL until those measurements are captured on the release revision with an exact isolated model and representative fixture, plus the advertised packaged variant.

## Commands and side effects

Focused tests were run with:

    .\app\server\.venv\Scripts\python.exe -m pytest -c app/server/pyproject.toml app/tests/unit/test_startup_validation.py app/tests/unit/test_database_initialization.py app/tests/unit/test_model_installation.py app/tests/unit/test_job_cancellation_semantics.py app/tests/unit/test_job_failure_semantics.py app/tests/unit/test_validation_job_semantics.py app/tests/unit/test_preparation_image_scanning.py app/tests/unit/test_training_stop_mechanism.py -q --basetemp=runtimes/cache/pytest-tmp/s50-s51-s53-20260925 -o cache_dir=runtimes/cache/pytest

The backend logger is initialized at import time against the canonical `app/resources/logs` path. This slice created the following four task-run logs; they were left untouched as requested:

- `XREPORT_20260925_124522.log` — created 2026-09-25 12:45:22 Europe/Rome, modified 12:45:26, 23,334 bytes; likely the focused pytest run.
- `XREPORT_20260925_124614.log` — created 12:46:14, modified 12:46:14, 881 bytes; likely first training race harness.
- `XREPORT_20260925_124637.log` — created 12:46:37, modified 12:46:38, 612 bytes; likely second training race harness.
- `XREPORT_20260925_124700.log` — created 12:47:00, modified 12:47:01, 459 bytes; likely final training race harness.

No listener was started. Disposable database/model probe roots were removed. Python __pycache__ directories from the test imports may remain; they were not removed because the primary task began making concurrent source/test changes in the shared checkout.
