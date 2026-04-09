# XREPORT User Manual

Last updated: 2026-04-08

This manual explains how to operate XREPORT in day-to-day usage.

## 1. Who This Is For

- Radiology/ML users running local report-generation workflows
- Technical operators validating datasets, training runs, and model outputs

## 2. Starting the Application

### 2.1 Windows (Local mode v1)
1. Run `XREPORT/start_on_windows.bat`.
2. Wait for runtime/dependency checks to complete.
3. Open the UI URL configured by `UI_HOST`/`UI_PORT` in `XREPORT/settings/.env`.

### 2.2 Windows (Local mode v2 packaged desktop)
1. Install and launch the packaged desktop app.
2. Wait for first-run backend initialization.
3. Use the embedded desktop UI.

### 2.3 macOS/Linux (manual)
1. Start backend:
```bash
uv run python -m uvicorn XREPORT.server.app:app --host 127.0.0.1 --port 8000
```
2. Start frontend preview:
```bash
cd XREPORT/client
npm run preview -- --host 127.0.0.1 --port 7861 --strictPort
```

## 3. Main User Journeys

### 3.1 Journey A: Prepare a dataset
1. Open **Dataset** page.
2. Load or upload a dataset.
3. Run preparation/processing.
4. Confirm dataset status and metadata before training.

Expected result:
- dataset is available in prepared/usable state for downstream training or validation.

### 3.2 Journey B: Train a model
1. Open **Training** page.
2. Choose dataset/checkpoint options and training parameters.
3. Start training.
4. Monitor live progress and metrics.
5. Stop or resume when needed.

Expected result:
- checkpoints are produced and listed for inference/validation.

### 3.3 Journey C: Generate reports (inference)
1. Open **Inference** page.
2. Select a checkpoint.
3. Submit image(s) for inference.
4. Poll job status until completed.
5. Review generated text outputs.

Expected result:
- draft reports are generated and available for review/export workflows.

### 3.4 Journey D: Validate quality
1. Start dataset validation or checkpoint evaluation from validation flows.
2. Wait for completion (polling-based).
3. Review quality metrics and generated validation artifacts.

Expected result:
- quality indicators available for model comparison and release decisions.

## 4. Primary Commands

### 4.1 Launch and runtime
- `XREPORT/start_on_windows.bat`
- `XREPORT/setup_and_maintenance.bat`

### 4.2 Backend/frontend manual run
- `uv run python -m uvicorn XREPORT.server.app:app --host <host> --port <port>`
- `cd XREPORT/client && npm run preview -- --host <host> --port <port> --strictPort`

### 4.3 Build packaged desktop (maintainers)
- `release/tauri/build_with_tauri.bat`

### 4.4 Tests
- `tests/run_tests.bat`
- `runtimes/.venv/Scripts/python.exe -m pytest tests -v --tb=short`

## 5. Usage Patterns and Best Practices

- Use a consistent dataset naming strategy to avoid confusion across runs.
- Validate dataset integrity before launching long training jobs.
- Track checkpoint purpose (baseline, tuned, experiment) with clear naming.
- Prefer one major long-running job at a time to reduce contention.
- Keep `XREPORT/settings/.env` aligned with your local host/port usage.

## 6. Key Features

- End-to-end workflow: dataset preparation -> training -> inference -> validation
- Long-running operations with start/poll/cancel behavior
- Dual runtime support:
  - Local mode (v1) web launcher
  - Local mode (v2) packaged desktop
- Embedded SQLite default with optional PostgreSQL mode

## 7. Troubleshooting Quick Guide

- UI not reachable:
  - check `UI_HOST` / `UI_PORT` in `XREPORT/settings/.env`
  - verify backend is running on `FASTAPI_HOST` / `FASTAPI_PORT`
- Jobs stay running too long:
  - poll status endpoint and check backend logs in `XREPORT/resources/logs`
- Missing artifacts/checkpoints:
  - confirm write permissions and paths under `XREPORT/resources`
- First run is slow:
  - expected when dependencies/runtime are being initialized

## 8. Data and Output Locations

- Runtime data root: `XREPORT/resources`
- Database file (SQLite mode): `XREPORT/resources/database.db`
- Checkpoints: `XREPORT/resources/checkpoints`
- Logs: `XREPORT/resources/logs`

## 9. Database Initialization

- SQLite mode (`database.embedded_database=true`):
  - On first startup only, if `XREPORT/resources/database.db` does not exist, the backend initializes the database automatically.
  - On later startups, initialization is skipped when the file is present.
- PostgreSQL mode (`database.embedded_database=false`):
  - Database initialization is not automatic at startup.
  - Run `XREPORT/setup_and_maintenance.bat`, choose option `1. Initialize database`, to execute `XREPORT/scripts/initialize_database.py`.
  - The same script also works for SQLite mode, but normally it is unnecessary because first-run SQLite initialization is automatic.
