# Commands And Locations

Last updated: 2026-06-03

## Primary Commands

### Launch And Maintenance

- `XREPORT/start_on_windows.bat`
- `XREPORT/setup_and_maintenance.bat`

### Manual Backend And Frontend

- `uv run python -m uvicorn XREPORT.server.app:app --host <host> --port <port>`
- `cd XREPORT/client && npm run preview -- --host <host> --port <port> --strictPort`

### Desktop Build

- `release/tauri/build_with_tauri.bat`

### Tests

- `tests/run_tests.bat`
- `runtimes/.venv/Scripts/python.exe -m pytest tests -v --tb=short`

## Usage Best Practices

- Use a consistent dataset naming strategy across runs.
- Validate dataset integrity before launching long training jobs.
- Track checkpoint purpose with clear naming such as baseline, tuned, or experiment.
- Prefer one major long-running job at a time to reduce contention.
- Keep `XREPORT/settings/.env` aligned with local host and port usage.

## Key Features

- end-to-end dataset preparation to training to inference to validation workflow
- long-running operations with start, poll, and cancel behavior
- dual runtime support through web launcher and packaged desktop modes
- embedded SQLite by default with optional PostgreSQL mode

## Data And Output Locations

- runtime data root: `XREPORT/resources`
- SQLite database file: `XREPORT/resources/database.db`
- checkpoints: `XREPORT/resources/checkpoints`
- logs: `XREPORT/resources/logs`
