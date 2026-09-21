# Commands And Locations

Last updated: 2026-09-21

## Primary Commands

### Launch And Maintenance

```powershell
.\start_on_windows.ps1 -Action LaunchDesktopDev
.\start_on_windows.ps1 -Action KillProcesses
.\start_on_windows.ps1 -Action BuildDesktopRelease -DesktopRuntime All -DesktopTarget All -Version 3.1.0
.\start_on_windows.ps1 -Action RemoveDesktopRelease
.\start_on_windows.ps1 -Action RemoveCheckpoints
.\start_on_windows.ps1 -Action RemoveAllData
```

The equivalent complete desktop wrapper from `app/desktop` is:

```powershell
npm run tauri:build
npm run tauri:build -- -DesktopRuntime Cuda -DesktopTarget All
```

Desktop release preparation uses `uv sync --frozen`, `npm ci`, and the pinned
Rust/Node/Python/uv versions. Generated `ui/` and `runtime.zip` inputs remain
ignored and are recreated for every release build.

Release artifacts are under `release/`. Desktop staging and Cargo output are
under `app/desktop/build` (including variant-specific `cargo-target` folders);
all are ignored.
From the interactive launcher, use the **DESKTOP RELEASE** section to create
or remove CPU/CUDA portable and MSI payloads individually, or select all four.
Interactive removal updates the selected variant manifests; the direct remove
action above removes the complete desktop release build output.
All removal actions require an interactive `[y/N]` confirmation and fail closed
when input is redirected. `RemoveAllData` targets only the configured local
resource data; external databases and tracked application files are preserved.
Packaged logs are `%LOCALAPPDATA%\XREPORT\data\logs\desktop-shell.log` and the
timestamped backend logs beside it. Readiness/session files in
`%LOCALAPPDATA%\XREPORT\data\state` are temporary and are removed at exit.

- `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1`
- `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action Launch`
- `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action RebuildFrontend`

### Desktop commands and locations

### Manual Backend And Frontend

- `uv run --project app/server python -m uvicorn server.app:app --app-dir app --host <host> --port <port>`
- `cd app/client && npm run preview -- --host <host> --port <port>`

### Tests

- `app/tests/run_tests.bat`
- `app/server/.venv/Scripts/python.exe -m pytest -c app/server/pyproject.toml app/tests -v --tb=short --basetemp runtimes/cache/pytest-tmp/manual -o "cache_dir=runtimes/cache/pytest"`
- `$env:PYTHONPATH = "app"; & ".\app\server\.venv\Scripts\python.exe" ".\app\scripts\validate_cxrmate_ed_sensitivity.py" --fixture-provenance "<approved source>" --fixture-deidentification "<approved de-identification statement>"`

The CXRMate-ED canary is cache-only and writes its real-inference evidence to
`assets/QA/inference_validation_runs/`. A failed canary is an expected,
explicit degraded result for research access; it must not be treated as a
passing validation receipt.

### Tier 0 validation

The hard-gated foundational campaign is documented in
[`validation_campaign_ledger.md`](../validation_campaign_ledger.md). Run its
current-head sequence in this order:

```powershell
uv sync --locked --extra test --python 3.14.7
Set-Location app/server
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pyright.exe .
Set-Location ../..
app\server\.venv\Scripts\python.exe -m pytest -c app/server/pyproject.toml app/tests/unit -q --basetemp runtimes/cache/pytest-tmp/tier0-unit -o "cache_dir=runtimes/cache/pytest"
Set-Location app/client
npm ci --no-audit --no-fund
npm run build
npm run lint
npm run test:unit
```

For the rendered startup gate, use the official launcher, then run the focused
test against the live pair:

```powershell
Set-Location ../..
.\start_on_windows.ps1 -Action Launch
$env:APP_TEST_FRONTEND_URL = "http://127.0.0.1:8003"
$env:APP_TEST_BACKEND_URL = "http://127.0.0.1:5003"
app\server\.venv\Scripts\python.exe -m pytest -c app/server/pyproject.toml app/tests/e2e/test_angular_ui.py::test_startup_gate_holds_inference_until_backend_health -q --basetemp runtimes/cache/pytest-tmp/tier0-s02 -o "cache_dir=runtimes/cache/pytest"
```

Durable campaign summaries belong under `assets/QA/validation_campaign/`;
transient caches, server logs, and generated bundles remain under
`runtimes/cache` unless a slice summary explicitly promotes them to evidence.

### Development cache locations

- all disposable caches: `runtimes/cache/{pytest,pytest-tmp,ruff,mypy,python,coverage,angular,uv,npm,pip,playwright-browsers,huggingface,torch,keras,matplotlib}`
- best-effort cleanup: `.\start_on_windows.ps1 -Action ClearCache`

Locked or administrator-protected cache files are reported and skipped during
cleanup; other cached artifacts continue to be removed.

### Alembic development workflow

Run these commands from `app/server` after the target database is available:

```powershell
uv run alembic -c alembic.ini current --check-heads
uv run alembic -c alembic.ini history
uv run alembic -c alembic.ini revision --autogenerate -m "describe schema change"
uv run alembic -c alembic.ini upgrade head
uv run alembic -c alembic.ini downgrade -1
```

Review and edit every autogenerated revision before committing it. The runtime
requires one linear head and never performs downgrades automatically.

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
- persistent model installations and lifecycle metadata: `<resource root>/models`
- transient model/tool caches: `<runtime root>/runtimes/cache` (or packaged `<data root>/runtimes/cache`)
- tokenizer resources: `<resource root>/tokenizers`
- report templates: `<resource root>/templates`
- logs: `<resource root>/logs`
