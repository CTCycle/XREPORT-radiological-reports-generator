# Packaging and Runtime Modes

Last updated: 2026-04-08

This document defines how XREPORT runtime settings and packaging modes work.

## 1. Runtime Strategy

Active runtime settings file:
- `XREPORT/settings/.env`

Non-runtime defaults (including DB settings):
- `XREPORT/settings/configurations.json`

## 2. Mode Definitions

### 2.1 Local mode (v1): web launcher
- Launcher: `XREPORT/start_on_windows.bat`
- Runs backend + frontend as local web services

### 2.2 Local mode (v2): packaged desktop
- Build helper: `release/tauri/build_with_tauri.bat`
- Produces Windows desktop artifacts under `release/windows`
- Packaged app runs local backend in background and serves the SPA

## 3. Environment Keys

| Key | Purpose |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host/port |
| `UI_HOST`, `UI_PORT` | Frontend bind host/port for local web mode |
| `VITE_API_BASE_URL` | Frontend API base path (keep `/api` for compatibility) |
| `RELOAD` | Uvicorn reload toggle in local dev |
| `OPTIONAL_DEPENDENCIES` | Installs optional Python dependencies in launcher flow |
| `MPLBACKEND`, `KERAS_BACKEND` | Runtime ML/plotting backend settings |

Database settings are read from JSON:
- `database.embedded_database` (`true` SQLite / `false` PostgreSQL)
- `database.engine`, `database.host`, `database.port`, `database.database_name`
- `database.username`, `database.password`, `database.ssl`, `database.ssl_ca`
- `database.connect_timeout`, `database.insert_batch_size`

Database initialization behavior:
- SQLite mode (`database.embedded_database=true`): startup initializes schema only when `XREPORT/resources/database.db` is missing.
- PostgreSQL mode (`database.embedded_database=false`): startup does not initialize the database; initialize manually from `XREPORT/setup_and_maintenance.bat` option `1` (runs `XREPORT/scripts/initialize_database.py`).

## 4. Local Mode (v1) Workflow

1. Verify/update `XREPORT/settings/.env`.
2. Start application:
   - `XREPORT\start_on_windows.bat`
3. Optional test run:
   - `tests\run_tests.bat`

## 5. Local Mode (v2) Workflow

1. Verify/update `XREPORT/settings/.env`.
2. Optional icon regeneration (when branding changes):
   - `cd XREPORT\client && npm run tauri:icon`
3. Build desktop artifacts:
   - `release\tauri\build_with_tauri.bat`
4. Collect release outputs:
   - `release/windows/installers` (preferred)
   - `release/windows/portable`

Optional cleanup:
- `cd XREPORT\client && npm run tauri:clean`
- or `XREPORT\setup_and_maintenance.bat` option `3. Clean desktop build artifacts`

## 6. Desktop Runtime Notes

- Packaged runtime tries to reuse an existing valid `runtimes/.venv` first.
- If no reusable runtime is found, runtime files are created under a writable runtime root.
- Dependency sync is lockfile-backed with `uv sync --frozen` and runs only when runtime venv is missing.
- First launch may still be slower due to heavy ML dependency resolution (`torch`, `torchvision`).
- Splash/status text is intentionally generic and should not expose absolute runtime paths.
- End users only need the produced installer/executable; Rust/Cargo is build-machine-only.

## 7. Deterministic Build Inputs

- Backend dependency graph: `runtimes/uv.lock` (staged to `uv.lock` during sync/build)
- Frontend dependency graph: `XREPORT/client/package-lock.json`
