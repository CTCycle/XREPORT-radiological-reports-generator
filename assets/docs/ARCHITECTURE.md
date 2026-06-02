# XREPORT Architecture

Last updated: 2026-05-31

## 1. System Architecture Overview

XREPORT is a local-first client/server system for radiological report generation and model lifecycle workflows.

- Frontend: React + TypeScript + Vite (`XREPORT/client`)
- Backend: FastAPI (`XREPORT/server`)
- Desktop wrapper: Tauri 2 (`XREPORT/client/src-tauri`)
- Persistence: SQLite by default, PostgreSQL optional (`XREPORT/server/repositories/database`)
- Long-running execution: job manager with start/poll/cancel contracts (`XREPORT/server/services/jobs.py`)

## 2. Directory and File Structure

The structure below is the implementation-relevant tree (dependency/build outputs such as `node_modules`, `dist`, `target`, and `__pycache__` are omitted).

```text
.
├─ pyproject.toml
├─ README.md
├─ runtimes/
│  ├─ uv.lock
│  └─ .venv/                        # created by launcher/uv
├─ assets/
│  └─ docs/
├─ release/
│  └─ tauri/
│     ├─ build_with_tauri.bat
│     └─ scripts/
│        ├─ clean-tauri-build.ps1
│        └─ export-windows-artifacts.ps1
├─ tests/
│  ├─ run_tests.bat
│  ├─ conftest.py
│  ├─ spaserver.py
│  ├─ unit/
│  └─ e2e/
└─ XREPORT/
   ├─ start_on_windows.bat
   ├─ setup_and_maintenance.bat
   ├─ settings/
   │  ├─ .env
   │  └─ configurations.json
   ├─ resources/
   │  ├─ checkpoints/
   │  ├─ logs/
   │  ├─ templates/
   │  ├─ tokenizers/
   │  └─ database.db                # SQLite mode
   ├─ scripts/
   │  └─ initialize_database.py
   ├─ server/
   │  ├─ app.py                     # FastAPI entrypoint
   │  ├─ api/
   │  │  ├─ upload.py
   │  │  ├─ preparation.py
   │  │  ├─ training.py
   │  │  ├─ validation.py
   │  │  └─ inference.py
   │  ├─ domain/                    # Pydantic request/response contracts
   │  ├─ services/                  # orchestration + job lifecycle
   │  ├─ repositories/
   │  │  ├─ database/               # SQLite/PostgreSQL backends and init
   │  │  ├─ queries/                # query adapters
   │  │  ├─ schemas/                # SQLAlchemy models
   │  │  └─ serialization/          # dataframe/domain persistence
   │  ├─ learning/                  # ML training/inference logic
   │  ├─ configurations/            # env + JSON settings loaders
   │  └─ common/
   │     ├─ constants.py
   │     └─ utils/
   └─ client/
      ├─ package.json
      ├─ vite.config.ts
      ├─ src/
      │  ├─ main.tsx                # React entrypoint
      │  ├─ App.tsx                 # route graph
      │  ├─ pages/
      │  │  ├─ DatasetPage.tsx
      │  │  ├─ TrainingPage.tsx
      │  │  ├─ InferencePage.tsx
      │  │  └─ DatasetValidationPage.tsx
      │  ├─ components/
      │  ├─ services/
      │  ├─ hooks/
      │  └─ types/
      └─ src-tauri/
         ├─ Cargo.toml
         ├─ tauri.conf.json
         └─ src/main.rs             # desktop runtime bootstrap
```

## 3. Entry Points

- Backend API entrypoint: `XREPORT/server/app.py`
- Frontend web entrypoint: `XREPORT/client/src/main.tsx`
- Frontend route composition: `XREPORT/client/src/App.tsx`
- Desktop entrypoint: `XREPORT/client/src-tauri/src/main.rs`
- Local launcher (Windows): `XREPORT/start_on_windows.bat`
- Desktop build flow (Windows): `release/tauri/build_with_tauri.bat`

## 4. Backend API Surface

All routers are mounted under `/api`.

### Upload
- `POST /api/upload/dataset`

### Preparation
- `GET /api/preparation/dataset/status`
- `GET /api/preparation/dataset/names`
- `GET /api/preparation/dataset/processed/names`
- `GET /api/preparation/dataset/metadata/{dataset_name}`
- `DELETE /api/preparation/dataset/{dataset_name}`
- `POST /api/preparation/images/validate`
- `POST /api/preparation/dataset/load`
- `POST /api/preparation/dataset/process`
- `GET /api/preparation/dataset/{dataset_name}/images/count`
- `GET /api/preparation/dataset/{dataset_name}/images/{index}`
- `GET /api/preparation/dataset/{dataset_name}/images/{index}/content`
- `GET /api/preparation/jobs/{job_id}`
- `DELETE /api/preparation/jobs/{job_id}`
- `GET /api/preparation/browse`

### Training
- `GET /api/training/checkpoints`
- `GET /api/training/checkpoints/{checkpoint}/metadata`
- `DELETE /api/training/checkpoints/{checkpoint}`
- `GET /api/training/status`
- `POST /api/training/start`
- `POST /api/training/resume`
- `GET /api/training/jobs/{job_id}`
- `DELETE /api/training/jobs/{job_id}`

### Validation
- `POST /api/validation/run`
- `POST /api/validation/checkpoint`
- `GET /api/validation/checkpoint/reports/{checkpoint}`
- `GET /api/validation/reports/{dataset_name}`
- `GET /api/validation/jobs/{job_id}`
- `DELETE /api/validation/jobs/{job_id}`

### Inference
- `GET /api/inference/checkpoints`
- `POST /api/inference/generate`
- `GET /api/inference/jobs/{job_id}`
- `DELETE /api/inference/jobs/{job_id}`

### Root behavior
- When `app/client/dist` is available: backend serves SPA files from the built frontend bundle
- Otherwise: `GET /` redirects to `/docs`

## 5. Layered Architecture and Responsibilities

### Endpoint layer (`XREPORT/server/api`)
- Parses transport concerns (multipart files, path/query/body parameters)
- Converts HTTP interactions into service calls
- Applies response models/status codes

### Service layer (`XREPORT/server/services`)
- Contains orchestration and business rules
- Starts and monitors long-running jobs
- Maps repository results into API/domain responses

### Repository layer (`XREPORT/server/repositories`)
- `database/*`: backend engine creation + DB initialization
- `schemas/*`: SQLAlchemy table definitions
- `queries/*`: data access adapters
- `serialization/*`: dataframe <-> persistence mapping and report/checkpoint storage

### Learning layer (`XREPORT/server/learning`)
- Model training/inference implementation details
- Trainer, scheduler, dataloader, callback and generator logic

### Frontend layer (`XREPORT/client/src`)
- `pages/*`: route-level flows
- `components/*`: reusable UI building blocks
- `services/*`: backend API integration and polling
- `hooks/*`: reusable async/job state patterns

## 6. Data Persistence and Storage

### Database backend selection
From `XREPORT/settings/configurations.json`:
- `database.embedded_database=true`: SQLite (`XREPORT/resources/database.db`)
- `database.embedded_database=false`: PostgreSQL (`database.engine`, host/port/db/user/pass/SSL settings)

### Initialization behavior
- Backend startup performs database initialization before serving requests
- SQLite: schema creation is ensured against the embedded database file
- PostgreSQL: database and schema initialization are executed from configured connection settings
- Additional startup validation ensures required resource directories exist and Tauri mode has a built frontend bundle available

### Persisted domains
Core persisted entities include:
- datasets + dataset records
- processing runs + training samples
- validation runs + text/image aggregates + pixel distributions
- checkpoints + checkpoint evaluations
- inference runs + generated reports

### Non-database persisted artifacts
- checkpoints and model artifacts under `XREPORT/resources/checkpoints` / `XREPORT/resources/models`
- logs under `XREPORT/resources/logs`
- templates and tokenizer resources under `XREPORT/resources/templates` and `XREPORT/resources/tokenizers`

## 7. Async vs Sync Behavior and Constraints

- Most backend operations are synchronous request handlers that delegate CPU-heavy/long-running work to background jobs (`threading.Thread`) via `JobManager`.
- Async API handlers are used where needed for I/O-facing operations:
  - multipart file reads (`upload`, `inference`)
  - async validation endpoints delegating to async service methods
- Long-running compute is not executed directly in request scope:
  - training is run through managed job execution and a process worker pipeline
  - validation/inference/preparation heavy tasks use job start + poll + cancel flow
- Database access is synchronous via SQLAlchemy engines/sessions (no async DB driver in current implementation).

## 8. Architectural Constraints

- The system is local-first and file-system aware (dataset browsing and local paths are part of the supported workflow).
- No authentication/authorization layer is implemented in the current API surface.
- Job progress UX is polling-based; no production WebSocket API is currently exposed by backend routes.
