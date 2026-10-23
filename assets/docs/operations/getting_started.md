# Getting Started

Last updated: 2026-10-23

This guidance is for radiology and ML users running local report-generation workflows, plus technical operators validating datasets, training runs, and model outputs.

## Startup Paths

### Windows Local Launcher

1. Run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1`.
2. Select **Launch application**.
3. Wait for runtime, dependency, database migration, build-freshness, and health checks to complete. The frontend builds only when its production output is absent or stale.
4. Use the browser opened at `http://FASTAPI_HOST:FASTAPI_PORT` from
   `settings/.env` (wildcard bind addresses open through loopback).

For direct launch without the menu, run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action Launch`. The launcher stays open while the app runs; press Ctrl+C to request a graceful stop.

### Optional XREPORT Studio Setup

XREPORT and XREPORT Studio use separate Python environments. The regular
XREPORT install does not install Studio-only packages. To prepare Studio ahead
of time, choose **Install / update Studio** or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action InstallStudio
```

Choose **Standard** for CPU Torch or **CUDA** for the pinned cu130 Torch stack
and a supported NVIDIA GPU/driver. To select a profile directly, add
`-StudioProfile Standard` or `-StudioProfile Cuda`.

Launch Studio with **Launch XREPORT Studio** or
`powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action LaunchStudio`.
On an interactive first launch, the launcher asks for a profile and installs
it before starting Studio. An unattended first launch defaults to Standard.
Subsequent launches reuse the installed profile; pass `-StudioProfile` to
switch it. The installation builds Studio's Angular frontend and keeps Studio
data under the existing `<resources>/studio` folder.

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

Start these two development processes in separate terminals.

Terminal 1, from the repository root:

```bash
uv run --project app/server --extra cuda python -m uvicorn server.app:app --app-dir app --host 127.0.0.1 --port 5003
```

Terminal 2, from the repository root:

```bash
cd app/client
npm run dev -- --host 127.0.0.1 --port 8003
```

The Angular development server proxies `/api` to FastAPI. Use `npm run build`
followed by `npm run preview` only to validate the built bundle locally or for
Tauri development; preview is not production hosting.

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
