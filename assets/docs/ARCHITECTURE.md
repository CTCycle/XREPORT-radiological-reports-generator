# XREPORT Architecture

## 1. System Overview

### 1.1 Purpose
XREPORT is an application for generating draft radiology reports from X-ray images.

### 1.2 Main workflows
- Dataset upload and preparation
- Model training and resume
- Inference on uploaded images
- Dataset validation and checkpoint evaluation

### 1.3 Runtime shape
- Frontend: React + Vite in `XREPORT/client`
- Backend: FastAPI in `XREPORT/server`
- Persistence: SQLite (embedded default) or PostgreSQL (external mode)
- Long-running work: centralized background job manager with polling endpoints

## 2. Repository Structure

### 2.1 Primary directories
| Path | Purpose |
|---|---|
| `XREPORT/client` | UI routes, components, styles, API service modules |
| `XREPORT/server/api` | FastAPI route modules |
| `XREPORT/server/domain` | Request/response models and job models |
| `XREPORT/server/services` | Domain services (jobs, processing, validation, evaluation) |
| `XREPORT/server/learning` | ML training and inference logic |
| `XREPORT/server/repositories` | Database backends, schema models, queries, serializers |
| `XREPORT/settings` | Active env and env templates |
| `XREPORT/resources` | Runtime data (DB file, checkpoints, models, logs) |
| `tests` | Unit, E2E, and backend verification tests |
| `runtimes` | Windows portable runtimes and `.venv` |

### 2.2 Backend entrypoints
- App composition: `XREPORT/server/app.py`
- Route modules:
  - `XREPORT/server/api/upload.py`
  - `XREPORT/server/api/preparation.py`
  - `XREPORT/server/api/training.py`
  - `XREPORT/server/api/inference.py`
  - `XREPORT/server/api/validation.py`

### 2.3 Frontend entrypoints
- Router shell: `XREPORT/client/src/App.tsx`
- Layout: `XREPORT/client/src/components/MainLayout.tsx`
- Top pages:
  - `XREPORT/client/src/pages/DatasetPage.tsx`
  - `XREPORT/client/src/pages/TrainingPage.tsx`
  - `XREPORT/client/src/pages/InferencePage.tsx`
  - `XREPORT/client/src/pages/DatasetValidationPage.tsx`

## 3. API Design

### 3.1 Route mounting
All routers are included twice in `app.py`:
- native path (for example `/training/start`)
- aliased `/api` path (for example `/api/training/start`)

This supports same-origin frontend calls in desktop mode while preserving direct backend paths.

### 3.2 Root behavior
- Desktop mode (`XREPORT_TAURI_MODE=true`) with packaged client available: backend serves SPA files from `XREPORT/client/dist`.
- Otherwise: `GET /` redirects to `/docs`.

### 3.3 Route inventory

#### Upload
| Method | Route |
|---|---|
| POST | `/upload/dataset` |

#### Preparation
| Method | Route |
|---|---|
| GET | `/preparation/dataset/status` |
| GET | `/preparation/dataset/names` |
| GET | `/preparation/dataset/processed/names` |
| GET | `/preparation/dataset/metadata/{dataset_name}` |
| DELETE | `/preparation/dataset/{dataset_name}` |
| POST | `/preparation/images/validate` |
| POST | `/preparation/dataset/load` |
| POST | `/preparation/dataset/process` |
| GET | `/preparation/dataset/{dataset_name}/images/count` |
| GET | `/preparation/dataset/{dataset_name}/images/{index}` |
| GET | `/preparation/dataset/{dataset_name}/images/{index}/content` |
| GET | `/preparation/jobs/{job_id}` |
| DELETE | `/preparation/jobs/{job_id}` |
| GET | `/preparation/browse` |

#### Training
| Method | Route |
|---|---|
| GET | `/training/checkpoints` |
| GET | `/training/checkpoints/{checkpoint}/metadata` |
| DELETE | `/training/checkpoints/{checkpoint:path}` |
| GET | `/training/status` |
| POST | `/training/start` |
| POST | `/training/resume` |
| GET | `/training/jobs/{job_id}` |
| DELETE | `/training/jobs/{job_id}` |
| POST | `/training/stop` |

#### Inference
| Method | Route |
|---|---|
| GET | `/inference/checkpoints` |
| POST | `/inference/generate` |
| GET | `/inference/jobs/{job_id}` |
| DELETE | `/inference/jobs/{job_id}` |

#### Validation
| Method | Route |
|---|---|
| POST | `/validation/run` |
| POST | `/validation/checkpoint` |
| GET | `/validation/checkpoint/reports/{checkpoint}` |
| GET | `/validation/reports/{dataset_name}` |
| GET | `/validation/jobs/{job_id}` |
| DELETE | `/validation/jobs/{job_id}` |

## 4. Frontend Integration

- Frontend calls `/api/...` endpoints.
- Vite proxy in `XREPORT/client/vite.config.ts` rewrites configured API base (default `/api`) to `http://<FASTAPI_HOST>:<FASTAPI_PORT>`.
- Current backend communication model is HTTP polling for long-running tasks.

## 5. Persistence Model

### 5.1 Database backend selection
- `DB_EMBEDDED=true`: SQLite via `SQLiteRepository`
- `DB_EMBEDDED=false`: PostgreSQL via `PostgresRepository`

### 5.2 SQLite location
SQLite DB file path resolves to:
- `XREPORT/resources/database.db`

### 5.3 Core tables
Defined in `XREPORT/server/repositories/schemas/models.py`:
- `datasets`
- `dataset_records`
- `processing_runs`
- `training_samples`
- `validation_runs`
- `validation_text_summary`
- `validation_image_stats`
- `validation_pixel_distribution`
- `checkpoints`
- `checkpoint_evaluations`
- `inference_runs`
- `inference_reports`

## 6. Background Jobs

- Global job manager: `XREPORT/server/services/jobs.py` (`job_manager` singleton)
- Job states: `pending`, `running`, `completed`, `failed`, `cancelled`
- Start endpoints return `job_id`; status is read through `GET /.../jobs/{job_id}`; cancellation via `DELETE /.../jobs/{job_id}`
- Training uses a monitored `ProcessWorker` subprocess for heavy ML execution

See `assets/docs/BACKGROUND_JOBS.md` for implementation details.

## 7. Runtime Modes

### 7.1 Local mode (v1)
- Launcher: `XREPORT/start_on_windows.bat`
- Runs backend + frontend web stack locally

### 7.2 Local mode (v2)
- Build helper: `release/tauri/build_with_tauri.bat`
- Packaged desktop executable starts local backend and serves SPA
- Runtime environment is resolved under a writable runtime root, with reuse of existing `runtimes/.venv` when available

See `assets/docs/PACKAGING_AND_RUNTIME_MODES.md` and `assets/docs/TAURI_PACKAGING_PLAYBOOK.md`.

## 8. Current Limitations

- No authentication/authorization layer
- Uploaded dataset content for `/upload/dataset` is held in process memory and is not durable across backend restarts
- Filesystem browsing endpoint (`/preparation/browse`) is oriented to local desktop usage
- Training websocket proxy entries exist in Vite config, but backend workflows are polling-based

