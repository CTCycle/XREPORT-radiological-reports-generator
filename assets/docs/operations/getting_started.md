# Getting Started

Last updated: 2026-09-22

This guidance is for radiology and ML users running local report-generation workflows, plus technical operators validating datasets, training runs, and model outputs.

## Startup Paths

### Windows Local Launcher

1. Run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1`.
2. Select **Launch application**.
3. Wait for runtime, dependency, database migration, build-freshness, and health checks to complete. The frontend builds only when its production output is absent or stale.
4. Use the browser opened at the URL configured by `UI_HOST` and `UI_PORT` in `settings/.env`.

For direct launch without the menu, run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action Launch`.

The first launch creates the configured SQLite database (or PostgreSQL database
when permitted) and applies all checked-in Alembic revisions. Subsequent
launches check `alembic_version` and apply only pending upgrades. To run the
same operation without launching the UI, choose **Initialize database** or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action InitializeDatabase
```

### macOS Or Linux Manual Flow

From the repository root, set the canonical disposable-cache environment once
before running the manual commands:

```bash
export XREPORT_CACHE_ROOT="$PWD/runtimes/cache"
export XDG_CACHE_HOME="$XREPORT_CACHE_ROOT"
export UV_CACHE_DIR="$XREPORT_CACHE_ROOT/uv"
export PIP_CACHE_DIR="$XREPORT_CACHE_ROOT/pip"
export NPM_CONFIG_CACHE="$XREPORT_CACHE_ROOT/npm"
export PLAYWRIGHT_BROWSERS_PATH="$XREPORT_CACHE_ROOT/playwright-browsers"
export PYTHONPYCACHEPREFIX="$XREPORT_CACHE_ROOT/python"
export MPLCONFIGDIR="$XREPORT_CACHE_ROOT/matplotlib"
```

1. Start the backend:

```bash
uv run --project app/server python -m uvicorn server.app:app --app-dir app --host 127.0.0.1 --port 5003
```

2. Start the frontend preview:

```bash
cd app/client
npm run build
npm run preview -- --host 127.0.0.1 --port 8003
```

## First Report-Generation Run

1. Open **Inference** and read the research-use warning.
2. Select a ready public model or a complete Custom XReport checkpoint. A public
   model that is not cached is downloaded and verified during its first
   **Generate** action; gated models also require the provider's access terms
   and credentials.
3. Add a de-identified radiograph or study, choose a supported generation
   profile, and submit the background job.
4. Review the editable report sections declared by the selected model and
   inspect the returned model, provider, revision, profile, and output metadata
   before copying or exporting the draft.

Generated reports are research-use drafts, not clinically approved reports.
