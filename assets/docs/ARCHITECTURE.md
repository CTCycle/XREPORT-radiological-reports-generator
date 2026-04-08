# XREPORT Architecture

Last updated: 2026-04-08

## 1. System Overview

### 1.1 Purpose
XREPORT generates draft radiology reports from X-ray images and supports dataset preparation, model training, inference, and validation workflows.

### 1.2 Runtime shape
- Frontend: React + Vite (`XREPORT/client`)
- Backend: FastAPI (`XREPORT/server`)
- Persistence: SQLite (default embedded mode) or PostgreSQL (external mode)
- Long-running work: centralized thread-based job manager with start/poll/cancel APIs

## 2. Repository Structure

| Path | Purpose |
|---|---|
| `XREPORT/client` | UI routes, components, API service modules, styles |
| `XREPORT/server/api` | FastAPI route modules (`upload`, `preparation`, `training`, `inference`, `validation`) |
| `XREPORT/server/domain` | Request/response and job models |
| `XREPORT/server/services` | Orchestration and domain services (jobs, processing, validation, evaluation) |
| `XREPORT/server/learning` | ML training and inference logic |
| `XREPORT/server/repositories` | DB backends, schema models, serializers, database bootstrap |
| `XREPORT/settings` | Runtime `.env` and JSON configuration |
| `XREPORT/resources` | Runtime data (database, checkpoints, models, logs) |
| `tests` | Unit, E2E, and verification tests |
| `runtimes` | Windows portable runtimes and optional runtime `.venv` |

## 3. Backend Composition

### 3.1 App entrypoint
- `XREPORT/server/app.py` creates the FastAPI app and mounts all routers.
- Each router is included twice:
  - native routes (for example `/training/start`)
  - `/api`-prefixed aliases (for example `/api/training/start`)

### 3.2 Root behavior
- If `XREPORT_TAURI_MODE=true` and `XREPORT/client/dist` exists, backend serves the SPA and static assets.
- Otherwise `GET /` redirects to `/docs`.

### 3.3 API route modules
- `upload.py`
- `preparation.py`
- `training.py`
- `inference.py`
- `validation.py`

## 4. Frontend Composition

### 4.1 Entry and routes
- App shell: `XREPORT/client/src/App.tsx`
- Main layout: `XREPORT/client/src/components/MainLayout.tsx`
- Route pages:
  - `/dataset` (`DatasetPage.tsx`)
  - `/training` (`TrainingPage.tsx`)
  - `/inference` (`InferencePage.tsx`)
  - `/dataset/validate/:datasetName` (`DatasetValidationPage.tsx`)

### 4.2 Backend communication
- Frontend uses `/api` semantics through service modules in `XREPORT/client/src/services`.
- Vite proxy in `XREPORT/client/vite.config.ts` rewrites API calls to `http://<FASTAPI_HOST>:<FASTAPI_PORT>`.
- Long-running operations use polling; websocket proxy entries exist for training/inference paths but operational flows are polling-based.

## 5. Persistence Model

### 5.1 Backend selection
From `XREPORT/settings/configurations.json`:
- `database.embedded_database=true`: SQLite backend
- `database.embedded_database=false`: PostgreSQL backend

### 5.2 SQLite location and schema
- SQLite file: `XREPORT/resources/database.db`
- Core tables are defined in `XREPORT/server/repositories/schemas/models.py`, including:
  - datasets and dataset records
  - processing runs and training samples
  - validation runs and aggregates
  - checkpoints and checkpoint evaluations
  - inference runs and inference reports

## 6. Background Job Architecture

- Global singleton: `job_manager` in `XREPORT/server/services/jobs.py`
- Execution model: daemon threads per job; cooperative cancellation via stop flag
- Training jobs can supervise/stop subprocess workers through service logic
- Job statuses: `pending`, `running`, `completed`, `failed`, `cancelled`
- Endpoint contract pattern:
  - start endpoint returns `job_id`
  - status endpoint polls `GET /.../jobs/{job_id}`
  - cancel endpoint uses `DELETE /.../jobs/{job_id}`

## 7. Runtime Modes

### 7.1 Local mode (v1)
- Launcher script: `XREPORT/start_on_windows.bat`
- Runs local backend + frontend web stack

### 7.2 Local mode (v2)
- Build helper: `release/tauri/build_with_tauri.bat`
- Generates desktop artifacts and packaged app
- Packaged app starts local backend and serves SPA from bundled frontend assets

## 8. Current Constraints

- No auth/authz layer in backend APIs.
- Upload flow is optimized for local desktop usage.
- Long-running UX is polling-first.
