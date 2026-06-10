# Runtime Startup

Last updated: 2026-06-03

## Windows Local Launcher

CMD:

```cmd
XREPORT\start_on_windows.bat
```

What it does:

- ensures portable Python, uv, and Node in `runtimes/`
- runs `uv sync`
- installs frontend dependencies and builds frontend when needed
- starts the backend with `uvicorn`
- starts the frontend with `npm run preview`

## Manual Backend And Frontend

PowerShell:

```powershell
uv run python -m uvicorn XREPORT.server.app:app --host 127.0.0.1 --port 5003
Set-Location XREPORT\client
npm run preview -- --host 127.0.0.1 --port 8003 --strictPort
```

Notes:

- Replace host and port using values from `XREPORT/settings/.env`.
- `VITE_API_BASE_URL` should remain `/api` for proxied local flow.

## Desktop Build And Packaging

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

The Tauri project itself now lives under `app\src-tauri`, but the client package still owns the frontend scripts and can invoke the desktop build with the updated relative config path.

Prerequisites:

- Rust and Cargo on the build machine
- runtime assets prepared by `XREPORT/start_on_windows.bat`

## Test Runtime

CMD:

```cmd
tests\run_tests.bat
```

Behavior:

- requires existing `runtimes\.venv`
- starts backend and frontend if they are not already running
- runs the pytest suite
