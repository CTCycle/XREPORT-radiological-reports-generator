# Commands And Locations

Last updated: 2026-10-28

## Primary Commands

### Launch And Maintenance

```powershell
.\start_on_windows.ps1 -Action Launch
.\start_on_windows.ps1 -Action LaunchDesktopDev
.\start_on_windows.ps1 -Action KillProcesses
.\start_on_windows.ps1 -Action BuildDesktopRelease -DesktopRuntime All -DesktopTarget All -Version 3.3.1
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

- `uv run --project app/server --extra cuda python -m uvicorn xreport_inference.app:app --app-dir app --host <host> --port <port>`
- `cd app/client && npm start -- --host <host> --port <port>` (Angular development server; `/api` is proxied to FastAPI)
- `cd app/client && npm run build && npm run preview -- --host <host> --port <port>` (local bundle validation or Tauri development only; not production hosting)

### Tests

- Full suite: `app/tests/run_tests.bat`
- Focused rendered browser suite: from `app/client`, run `npm run test:e2e`
- Built frontend server: from `app/client`, run `npm run test:built-server`
- `$env:PYTHONPATH = "app"; & ".\app\server\.venv\Scripts\python.exe" ".\app\scripts\validate_inference_model.py" --help`

For direct PowerShell pytest runs, resolve the cache root from the repository
root so pytest's configuration directory cannot redirect the relative
`cache_dir` into `app/server/runtimes/cache`:

```powershell
$cacheRoot = (Resolve-Path .\runtimes\cache).Path
$pytestCache = Join-Path $cacheRoot 'pytest'
$pytestTemp = Join-Path $cacheRoot 'pytest-tmp\manual'
& .\app\server\.venv\Scripts\python.exe -m pytest `
  -c .\app\server\pyproject.toml .\app\tests `
  -v --tb=short --basetemp $pytestTemp -o "cache_dir=$pytestCache"
```

The former CXRMate-ED sensitivity wrapper has been retired with the model. The
generic validator is for active catalogue entries only; a model that fails
qualification is removed from the selectable catalogue rather than retained
as degraded research access.

### Tier 0 validation

The hard-gated foundational campaign is documented in
[`validation_campaign_ledger.md`](../validation_campaign_ledger.md). Run its
current-head sequence in this order:

```powershell
uv sync --locked --extra cuda --extra studio --extra test --python 3.14.7
Set-Location app/server
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pyright.exe .
Set-Location ../..
$cacheRoot = (Resolve-Path .\runtimes\cache).Path
$pytestCache = Join-Path $cacheRoot 'pytest'
$pytestTemp = Join-Path $cacheRoot 'pytest-tmp\tier0-unit'
& .\app\server\.venv\Scripts\python.exe -m pytest `
  -c .\app\server\pyproject.toml .\app\tests\unit `
  -q --basetemp $pytestTemp -o "cache_dir=$pytestCache"
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
$repoRoot = (Resolve-Path .).Path
$env:XREPORT_E2E_SCREENSHOT_DIR = Join-Path $repoRoot 'assets/QA/validation_campaign/tier-0/s02'
$cacheRoot = (Resolve-Path .\runtimes\cache).Path
$pytestCache = Join-Path $cacheRoot 'pytest'
$pytestTemp = Join-Path $cacheRoot 'pytest-tmp\tier0-s02'
& .\app\server\.venv\Scripts\python.exe -m pytest `
  -c .\app\server\pyproject.toml `
  .\app\tests\e2e/test_angular_ui.py::test_startup_gate_holds_inference_until_backend_health `
  .\app\tests\e2e/test_angular_ui.py::test_startup_timing_retry_and_post_ready_feature_error `
  -q --basetemp $pytestTemp -o "cache_dir=$pytestCache"
```

Store QA screenshots, logs, and working reports under `assets/QA/`. Record the
result, source revision, scope, and remaining limit in the existing status or
campaign ledger instead of adding a separate summary for each slice. Track an
artifact only when project tooling requires it; the approved release manifest
is the current example. Disposable caches and generated bundles belong under
`runtimes/cache`.

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
- SQLite and PostgreSQL backends, selected from Settings → Database

## Data And Output Locations

- runtime data root: `data` by default; change it in Settings → Storage (applied on restart) or override with `XREPORT_RESOURCES_DIR`
- SQLite database file: `<resource root>/database.db`
- checkpoints: `<resource root>/checkpoints`
- persistent model installations and lifecycle metadata: `<resource root>/models`
- transient model/tool caches: `<runtime root>/runtimes/cache` (or packaged `<data root>/runtimes/cache`)
- tokenizer resources: `<resource root>/tokenizers`
- report templates: `<resource root>/templates`
- logs: `<resource root>/logs`
