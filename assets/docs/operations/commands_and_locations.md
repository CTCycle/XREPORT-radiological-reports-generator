# Commands And Locations

Last updated: 2026-08-19

## Primary Commands

### Launch And Maintenance

- `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1`
- `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action Launch`
- `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action RebuildFrontend`

### Manual Backend And Frontend

- `uv run --project app/server python -m uvicorn server.app:app --app-dir app --host <host> --port <port>`
- `cd app/client && npm run preview -- --host <host> --port <port>`

### Tests

- `app/tests/run_tests.bat`
- `app/server/.venv/Scripts/python.exe -m pytest app/tests -v --tb=short --basetemp .pytest-tmp`

## Usage Best Practices

- Use a consistent dataset naming strategy across runs.
- Validate dataset integrity before launching long training jobs.
- Track checkpoint purpose with clear naming such as baseline, tuned, or experiment.
- Prefer one major long-running job at a time to reduce contention.
- Keep `settings/.env` aligned with local host and port usage.

## Key Features

- end-to-end dataset preparation to training to inference to validation workflow
- long-running operations with start, poll, and cancel behavior
- local web runtime with a consolidated Windows launcher and maintenance menu
- SQLite by default with optional PostgreSQL mode selected by `EMBEDDED_DATABASE`

## Data And Output Locations

- runtime data root: `app/resources` by default; override with `XREPORT_RESOURCES_DIR`
- SQLite database file: `<resource root>/database.db`
- checkpoints: `<resource root>/checkpoints`
- model cache/artifacts: `<resource root>/models`
- tokenizer resources: `<resource root>/tokenizers`
- report templates: `<resource root>/templates`
- logs: `<resource root>/logs`
