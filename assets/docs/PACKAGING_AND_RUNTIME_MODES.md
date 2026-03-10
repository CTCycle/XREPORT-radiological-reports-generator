# XREPORT Packaging and Runtime Modes

## 1. Strategy

XREPORT uses one active runtime file: `XREPORT/settings/.env`.

- Local mode (v1): run web stack directly on host via `start_on_windows.bat`.
- Local mode (v2): build and distribute a packaged Tauri desktop release.
- Cloud mode: run with Docker (`backend` + `frontend`).
- Mode switching: replace values in `XREPORT/settings/.env` only.

## 2. Runtime Profiles

- `XREPORT/settings/.env.local.example`: local web defaults for Local mode (v1).
- `XREPORT/settings/.env.local.tauri.example`: packaged desktop defaults for Local mode (v2).
- `XREPORT/settings/.env.cloud.example`: cloud defaults for Docker deployment.
- `XREPORT/settings/.env`: active profile used by launchers, tests, and Docker runtime env loading.
- `XREPORT/settings/configurations.json`: non-runtime defaults only (global seed + job polling interval).

## 3. Required Environment Keys

| Key | Purpose |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host/port in local mode; backend host-published port in cloud compose env interpolation. |
| `UI_HOST`, `UI_PORT` | Frontend bind host/port (local v1) and host-published UI port (cloud compose env interpolation). |
| `VITE_API_BASE_URL` | Frontend API base path. Must stay `/api` for same-origin proxying and packaged desktop compatibility. |
| `RELOAD` | Enables backend reload in local development when `true`. |
| `OPTIONAL_DEPENDENCIES` | Enables optional test dependencies in launcher flow. |
| `DB_EMBEDDED` | `true` uses SQLite; `false` uses external DB settings. |
| `DB_ENGINE`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | External DB connection settings used when `DB_EMBEDDED=false`. |
| `DB_SSL`, `DB_SSL_CA` | External DB TLS settings. |
| `DB_CONNECT_TIMEOUT`, `DB_INSERT_BATCH_SIZE` | DB connection and write-batching runtime settings. |
| `MPLBACKEND`, `KERAS_BACKEND` | Runtime backend settings for plotting and ML stack. |
| `ALLOW_LOCAL_FILESYSTEM_ACCESS` | Enables server-side local filesystem helper endpoints (`/preparation/browse`, path validation/load, image file serving). Use `true` for local modes and `false` for cloud mode. |

## 4. Local Mode (v1: Web Launcher)

1. Copy local v1 profile values into active env:
   - `copy /Y XREPORT\settings\.env.local.example XREPORT\settings\.env`
2. Start application:
   - `XREPORT\start_on_windows.bat`
3. Run tests (optional):
   - `tests\run_tests.bat`

## 5. Local Mode (v2: Packaged Tauri Desktop)

1. Copy local v2 profile values into active env:
   - `copy /Y XREPORT\settings\.env.local.tauri.example XREPORT\settings\.env`
2. If desktop branding changed, regenerate desktop icons from the shared favicon source:
   - `cd XREPORT\client && npm run tauri:icon`
3. Build desktop package:
   - `release\tauri\build_with_tauri.bat`
4. Distribute installer/executable artifacts from:
   - `release/windows/installers` (preferred for end users)
   - `release/windows/portable` (app executable plus required runtime resources)
5. Clean previous desktop build residue (optional):
   - `cd XREPORT\client && npm run tauri:clean`
   - or use option `3. Clean desktop build artifacts` in `XREPORT\setup_and_maintenance.bat`

Runtime behavior:
- The packaged desktop executable starts a local backend process in the background.
- The desktop bootstrap discovers packaged workspace roots and prefers a valid workspace that already has `.venv\Scripts\python.exe`.
- When no reusable `.venv` is found, runtime files are created in a writable location: packaged workspace root when writable, otherwise `%LOCALAPPDATA%\com.xreport.desktop\runtime` (installer-safe fallback).
- `uv sync` runs only when `<runtime-root>\.venv\Scripts\python.exe` is missing and uses `UV_PROJECT_ENVIRONMENT` plus `--frozen` for lockfile-backed sync.
- The desktop bootstrap stores uv cache in `<runtime-root>\.uv-cache`.
- Backend starts uvicorn on `FASTAPI_HOST:FASTAPI_PORT` from the resolved runtime `.venv`.
- Desktop window loads `http://<FASTAPI_HOST>:<FASTAPI_PORT>/` when the API is reachable.
- First launch can still be long because model dependencies (for example `torch`/`torchvision`) may need to be resolved; this is independent from v1 runtime state.
- `uv sync` startup phase has a 15-minute timeout and surfaces an error screen instead of waiting indefinitely.
- Startup splash synchronization text is intentionally generic and does not expose absolute filesystem paths.
- End users run the shipped installer/`.exe` and do not need Rust/Cargo.
- Desktop icon assets are sourced from `XREPORT/client/public/favicon.png` and regenerated via `npm run tauri:icon`.
- The repository intentionally keeps only desktop icon outputs in `XREPORT/client/src-tauri/icons`; generated `android` and `ios` folders are removed after regeneration.
- Windows Explorer can cache stale icons for unchanged exe names/paths; if a rebuilt app appears unchanged, verify the binary or refresh the icon cache before assuming packaging failed.
- Do not delete `XREPORT/client/src-tauri`; it contains source/config files, not just build output.

## 6. Cloud Mode (Docker)

1. Copy cloud profile values into active env:
   - `copy /Y XREPORT\settings\.env.cloud.example XREPORT\settings\.env`
2. Build images (determinism check):
   - `docker compose --env-file XREPORT/settings/.env build --no-cache`
3. Start containers:
   - `docker compose --env-file XREPORT/settings/.env up -d`
4. Stop containers:
   - `docker compose --env-file XREPORT/settings/.env down`

Cloud topology:
- `backend`: FastAPI/Uvicorn container.
- `frontend`: Nginx container serving static frontend.
- `/api` on frontend origin is reverse-proxied to backend (`backend:8000`).
- Backend port is container-internal only (`expose: 8000`), not published to the host in cloud mode.

## 7. Deterministic Build Notes

- Backend dependency graph is lockfile-backed via `uv.lock` and installed with `uv sync --frozen`.
- Frontend dependency graph is lockfile-backed via `XREPORT/client/package-lock.json` and installed with `npm ci`.
- Docker base images are pinned to explicit tags in `docker/backend.Dockerfile` and `docker/frontend.Dockerfile`.
