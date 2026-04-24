# Runtime Modes

Last updated: 2026-04-24

## 1. Supported Modes

## 1.1 Local Development (Web mode)

- Backend: FastAPI (`XREPORT/server/app.py`)
- Frontend: Vite preview/dev server (`XREPORT/client`)
- Typical operator flow on Windows uses `XREPORT/start_on_windows.bat`.

## 1.2 Desktop Runtime (Tauri mode)

- Desktop shell: `XREPORT/client/src-tauri/src/main.rs`
- Bundled with Tauri config from `XREPORT/client/src-tauri/tauri.conf.json`.
- Desktop app starts local backend process and loads the web UI from local HTTP.

## 1.3 Containerized runtime

- Not implemented in current codebase (no maintained Docker runtime entrypoint detected).

## 2. Startup Procedures

## 2.1 Windows local launcher (recommended)

CMD:
```cmd
XREPORT\start_on_windows.bat
```

What it does:
- ensures portable Python/uv/Node in `runtimes/`
- runs `uv sync`
- installs frontend dependencies/builds frontend as needed
- starts backend (`uvicorn`) and frontend (`npm run preview`)

## 2.2 Manual backend + frontend (cross-platform developer flow)

PowerShell:
```powershell
uv run python -m uvicorn XREPORT.server.app:app --host 127.0.0.1 --port 5003
Set-Location XREPORT\client
npm run preview -- --host 127.0.0.1 --port 8003 --strictPort
```

Notes:
- Replace host/port using values from `XREPORT/settings/.env`.
- `VITE_API_BASE_URL` should remain `/api` for proxied local flow.

## 2.3 Desktop build and packaging (Windows)

CMD:
```cmd
release\tauri\build_with_tauri.bat
```

Equivalent client-local commands:
```cmd
cd XREPORT\client
npm run tauri:build
npm run tauri:export:windows
```

Prerequisites:
- Rust/Cargo on build machine
- runtime assets prepared by `XREPORT/start_on_windows.bat`

## 2.4 Test runtime

CMD:
```cmd
tests\run_tests.bat
```

Behavior:
- requires existing `runtimes\.venv`
- starts backend/frontend if not already running
- runs pytest suite

## 3. Configuration Differences

## 3.1 Shared configuration sources

- Env overrides: `XREPORT/settings/.env`
- Static configuration: `XREPORT/settings/configurations.json`

## 3.2 Key environment variables

- `FASTAPI_HOST`, `FASTAPI_PORT`
- `UI_HOST`, `UI_PORT`
- `VITE_API_BASE_URL` (expected `/api`)
- `RELOAD`
- `OPTIONAL_DEPENDENCIES`
- `MPLBACKEND`, `KERAS_BACKEND`

## 3.3 Database mode switch

From `configurations.json`:
- `database.embedded_database=true` -> SQLite
- `database.embedded_database=false` -> PostgreSQL

Initialization differences:
- SQLite: auto schema creation only when DB file is missing
- PostgreSQL: explicit manual init via `XREPORT/scripts/initialize_database.py` (through maintenance menu option 1)

## 4. Interoperability

- Frontend talks to backend via `/api` paths.
- Vite dev/preview proxies `/api` to `http://FASTAPI_HOST:FASTAPI_PORT`.
- Tauri desktop starts backend locally, waits for TCP readiness, then redirects the desktop window to backend root URL.
- Backend in Tauri mode serves packaged SPA assets when `XREPORT_TAURI_MODE=true` and `client/dist` is available.

## 5. Limitations and Constraints

- Desktop mode is implemented for Windows in current runtime bootstrap.
- First launch can be slow due to dependency synchronization (Torch stack and related ML packages).
- Long-running ML tasks are job-based and poll-driven; no production WebSocket API routes are currently exposed.
- Local filesystem browsing is feature-gated by configuration (`features.allow_local_filesystem_access`).

## 6. Deployment Notes

- Current deployment/distribution implementation is Windows-focused:
  - installer/portable artifacts exported to `release/windows`.
- Build and package flow:
  1. Prepare runtimes via launcher
  2. Build Tauri app
  3. Export artifacts with `tauri:export:windows`
- Runtime lock consistency for packaged workflow is anchored to `runtimes/uv.lock`.
