# XREPORT Packaging and Runtime Modes

## 1. Strategy

XREPORT uses one active runtime file: `XREPORT/settings/.env`.

- Local mode: run directly on host without Docker.
- Cloud mode: run with Docker (`backend` + `frontend`).
- Mode switching: replace values in `XREPORT/settings/.env` only.
- No business-logic branching is required for runtime mode changes.

## 2. Runtime Profiles

- `XREPORT/settings/.env.local.example`: local defaults (loopback host values, embedded DB).
- `XREPORT/settings/.env.cloud.example`: cloud defaults (container bind host values, external DB).
- `XREPORT/settings/.env`: active profile used by launcher, tests, and Docker runtime env loading.
- `XREPORT/settings/configurations.json`: non-runtime defaults only (no database runtime settings).

## 3. Required Environment Keys

| Key | Purpose |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host/port in local mode; backend host-published port in cloud compose env interpolation. |
| `UI_HOST`, `UI_PORT` | Frontend bind host/port (local) and host-published UI port (cloud compose env interpolation). |
| `VITE_API_BASE_URL` | Frontend API base path. Must stay `/api` for same-origin proxying. |
| `RELOAD` | Enables backend reload in local development when `true`. |
| `OPTIONAL_DEPENDENCIES` | Enables optional test dependencies in local launcher flow. |
| `DB_EMBEDDED` | `true` uses SQLite; `false` uses external DB settings. |
| `DB_ENGINE`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | External DB connection settings used when `DB_EMBEDDED=false`. |
| `DB_SSL`, `DB_SSL_CA` | External DB TLS settings. |
| `DB_CONNECT_TIMEOUT`, `DB_INSERT_BATCH_SIZE` | DB connection and write-batching runtime settings. |
| `MPLBACKEND`, `KERAS_BACKEND` | Runtime backend settings for plotting and ML stack. |

## 4. Local Mode (Default)

1. Copy local profile values into active env:
   - `copy /Y XREPORT\settings\.env.local.example XREPORT\settings\.env`
2. Start application:
   - `XREPORT\start_on_windows.bat`
3. Run tests (optional):
   - `tests\run_tests.bat`

Local mode does not require Docker.

## 5. Cloud Mode (Docker)

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

## 6. Deterministic Build Notes

- Backend dependency graph is lockfile-backed via `uv.lock` and installed with `uv sync --frozen`.
- Frontend dependency graph is lockfile-backed via `XREPORT/client/package-lock.json` and installed with `npm ci`.
- Docker base images are pinned to explicit tags in `docker/backend.Dockerfile` and `docker/frontend.Dockerfile`.
