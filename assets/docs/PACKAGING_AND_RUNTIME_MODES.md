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
2. Build desktop package:
   - `XREPORT\build_with_tauri.bat`
3. Distribute installer/executable artifacts from:
   - `release/windows/installers` (preferred for end users)
   - `release/windows/portable` (raw app executable)
4. Clean previous desktop build residue (optional):
   - `cd XREPORT\client && npm run tauri:clean`
   - or use option `3. Clean desktop build artifacts` in `XREPORT\setup_and_maintenance.bat`

Runtime behavior:
- The packaged desktop executable starts a local backend process in the background.
- Backend starts uvicorn on `FASTAPI_HOST:FASTAPI_PORT`.
- Desktop window loads `http://<FASTAPI_HOST>:<FASTAPI_PORT>/`.
- End users run the shipped installer/`.exe` and do not need Rust/Cargo.
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


