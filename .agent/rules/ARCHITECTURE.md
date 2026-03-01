# XREPORT Architecture

## 1. High-Level Overview

### 1.1 Purpose and scope
XREPORT is a web application for generating radiology report drafts from X-ray images. It provides an end-to-end workflow:
- dataset upload and preparation
- model training and resume
- inference on uploaded images
- dataset validation and checkpoint evaluation

### 1.2 System shape
- Frontend: React + Vite app in `XREPORT/client`
- Backend: FastAPI service in `XREPORT/server`
- Persistence: SQLite (embedded default) or PostgreSQL (external mode), via SQLAlchemy repositories
- Long-running tasks: centralized job system with polling endpoints

### 1.3 Runtime assumptions
- Python: `>=3.14` (`pyproject.toml`)
- Backend startup entrypoint: `XREPORT/server/app.py`
- Active runtime env file: `XREPORT/settings/.env`
- Non-runtime defaults: `XREPORT/settings/configurations.json` (global seed + job polling interval)
- Main runtime data path: `XREPORT/resources` (checkpoints, models, logs, DB, runtimes)

## 2. Codebase Structure

### 2.1 Primary directories
| Path | Purpose |
|---|---|
| `XREPORT/client` | React UI, route pages, API service modules |
| `XREPORT/server/routes` | FastAPI route modules (upload, preparation, training, validation, inference) |
| `XREPORT/server/entities` | Pydantic request/response models and job state dataclass |
| `XREPORT/server/services` | Domain services (jobs, processing, validation, evaluation) |
| `XREPORT/server/learning` | Training/inference ML logic and callbacks |
| `XREPORT/server/repositories` | Database backends, schema models, queries, serializers |
| `XREPORT/settings` | `.env` profiles and JSON configuration |
| `XREPORT/resources` | Runtime artifacts and persistent assets |
| `tests` | unit, e2e, and verification tests |

### 2.2 Backend module organization
- App composition: `XREPORT/server/app.py`
- Routes:
  - `XREPORT/server/routes/upload.py`
  - `XREPORT/server/routes/preparation.py`
  - `XREPORT/server/routes/training.py`
  - `XREPORT/server/routes/inference.py`
  - `XREPORT/server/routes/validation.py`
- Entities:
  - `XREPORT/server/entities/training.py`
  - `XREPORT/server/entities/inference.py`
  - `XREPORT/server/entities/validation.py`
  - `XREPORT/server/entities/jobs.py`
- DB layer:
  - `XREPORT/server/repositories/database/backend.py`
  - `XREPORT/server/repositories/schemas/models.py`
  - `XREPORT/server/repositories/serialization/data.py`
  - `XREPORT/server/repositories/serialization/model.py`

### 2.3 Frontend module organization
- Router shell: `XREPORT/client/src/App.tsx`, `XREPORT/client/src/components/MainLayout.tsx`
- Top-level pages:
  - `XREPORT/client/src/pages/DatasetPage.tsx`
  - `XREPORT/client/src/pages/TrainingPage.tsx`
  - `XREPORT/client/src/pages/InferencePage.tsx`
  - `XREPORT/client/src/pages/DatasetValidationPage.tsx`
- API clients:
  - `XREPORT/client/src/services/trainingService.ts`
  - `XREPORT/client/src/services/inferenceService.ts`
  - `XREPORT/client/src/services/validationService.ts`

## 3. Backend API

### 3.1 API style
- JSON REST endpoints via FastAPI.
- Long operations return `job_id` and use polling (`GET /.../jobs/{job_id}`).
- Root route redirects to Swagger docs:
  - `GET /` -> `/docs`

### 3.2 Route inventory

#### Upload
| Method | Route | Description |
|---|---|---|
| POST | `/upload/dataset` | Upload CSV/XLSX metadata file into temporary in-memory upload state |

#### Preparation
| Method | Route | Description |
|---|---|---|
| GET | `/preparation/dataset/status` | Source dataset row availability status |
| GET | `/preparation/dataset/names` | Source dataset list with row counts/report flag |
| GET | `/preparation/dataset/processed/names` | Processed dataset list (latest processing runs) |
| GET | `/preparation/dataset/metadata/{dataset_name}` | Latest processing metadata for dataset |
| DELETE | `/preparation/dataset/{dataset_name}` | Delete dataset by name |
| POST | `/preparation/images/validate` | Validate image folder path |
| POST | `/preparation/dataset/load` | Match uploaded records with image folder and persist source records |
| POST | `/preparation/dataset/process` | Start dataset processing job |
| GET | `/preparation/dataset/{dataset_name}/images/count` | Dataset image count |
| GET | `/preparation/dataset/{dataset_name}/images/{index}` | Dataset image metadata by 1-based index |
| GET | `/preparation/dataset/{dataset_name}/images/{index}/content` | Serve image file content |
| GET | `/preparation/jobs/{job_id}` | Preparation job status |
| DELETE | `/preparation/jobs/{job_id}` | Cancel preparation job |
| GET | `/preparation/browse` | Server-side directory browser (Windows drive-oriented) |

#### Training
| Method | Route | Description |
|---|---|---|
| GET | `/training/checkpoints` | List checkpoints (config metadata only) |
| GET | `/training/checkpoints/{checkpoint}/metadata` | Load checkpoint metadata/config/session |
| DELETE | `/training/checkpoints/{checkpoint}` | Delete checkpoint directory |
| GET | `/training/status` | Current training dashboard state |
| POST | `/training/start` | Start training job |
| POST | `/training/resume` | Resume training from checkpoint |
| GET | `/training/jobs/{job_id}` | Training job status |
| DELETE | `/training/jobs/{job_id}` | Cancel training job |
| POST | `/training/stop` | Legacy stop endpoint (kept for backward compatibility) |

#### Inference
| Method | Route | Description |
|---|---|---|
| GET | `/inference/checkpoints` | List inference checkpoints |
| POST | `/inference/generate` | Start inference job for uploaded images |
| GET | `/inference/jobs/{job_id}` | Inference job status |
| DELETE | `/inference/jobs/{job_id}` | Cancel inference job |

#### Validation and checkpoint evaluation
| Method | Route | Description |
|---|---|---|
| POST | `/validation/run` | Start dataset validation job |
| POST | `/validation/checkpoint` | Start checkpoint evaluation job |
| GET | `/validation/checkpoint/reports/{checkpoint}` | Get persisted checkpoint evaluation report |
| GET | `/validation/reports/{dataset_name}` | Get persisted dataset validation report |
| GET | `/validation/jobs/{job_id}` | Validation/evaluation job status |
| DELETE | `/validation/jobs/{job_id}` | Cancel validation/evaluation job |

### 3.3 Schemas and response models
- Request/response models live under `XREPORT/server/entities/*.py`.
- Job payload models are shared in `XREPORT/server/entities/jobs.py`.
- Route modules use class-based endpoint objects and register paths with `add_api_route`.

### 3.4 Auth and access control
- No authentication/authorization layer is currently implemented.

### 3.5 Error handling
- Errors are returned primarily via `HTTPException` with status codes 400/404/409/422/500.
- Job failures surface via job status payload (`status=failed`, `error=...`).

## 4. Frontend Architecture

### 4.1 UI navigation
Sidebar navigation currently includes:
- Dataset
- Training
- Inference

Routes:
- `/dataset`
- `/training`
- `/inference`
- `/dataset/validate/:datasetName`

### 4.2 Frontend state model
- Page-level state is centralized through `AppStateContext.tsx`.
- Long-running workflows persist active job references in local storage through `usePersistedRecord`.
- API modules in `src/services` wrap all backend calls and expose polling helpers.

### 4.3 API integration
- Frontend uses `/api/...` calls.
- Vite proxy configuration in `XREPORT/client/vite.config.ts` rewrites `/api` to backend `FASTAPI_HOST:FASTAPI_PORT`.
- Poll intervals can be controlled by backend response field `poll_interval`.

## 5. Persistence Model

### 5.1 Database backends
- Embedded mode: SQLite via `SQLiteRepository`
- External mode: PostgreSQL via `PostgresRepository`
- Backend selection is driven by `DB_EMBEDDED` and DB env vars.

### 5.2 Canonical schema tables
Defined in `XREPORT/server/repositories/schemas/models.py` and constants in `XREPORT/server/common/constants.py`:
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

### 5.3 Serialization layer
`DataSerializer` is the main persistence orchestrator for:
- source dataset upsert and retrieval
- processing run + training samples save/load
- validation report save/load
- checkpoint evaluation report save/load
- generated inference reports save

## 6. Background Job Architecture

### 6.1 Job manager core
- Singleton `job_manager` in `XREPORT/server/services/jobs.py`.
- Starts daemon threads, tracks state, supports cancellation flagging.
- Merges partial `update_result(...)` payloads with final runner result.

### 6.2 Job lifecycle
- `pending` -> `running` -> terminal (`completed`, `failed`, `cancelled`)
- Polling endpoints expose progress and latest result payload.

### 6.3 Training special case
- Training route starts a job thread that supervises a `ProcessWorker` subprocess.
- Route-level monitor loop handles:
  - callback message polling
  - progress/result updates
  - cooperative stop
  - forced termination on stop timeout

## 7. Main Application Flows

### 7.1 Dataset ingestion and preparation
1. Upload CSV/XLSX via `/upload/dataset`.
2. Validate folder path via `/preparation/images/validate`.
3. Load + match records/images via `/preparation/dataset/load`.
4. Process dataset via `/preparation/dataset/process` (sanitize, tokenize, split, persist training samples/metadata).

### 7.2 Training and resume
1. Choose processed dataset and post `/training/start` (or `/training/resume`).
2. Poll `/training/jobs/{job_id}` and `/training/status`.
3. Artifacts/checkpoints are persisted under `XREPORT/resources/checkpoints`.

### 7.3 Inference
1. Upload images + checkpoint + generation mode to `/inference/generate`.
2. Poll `/inference/jobs/{job_id}` for progress and report payload.
3. Generated outputs are persisted to inference tables.

### 7.4 Validation and checkpoint evaluation
1. Dataset validation: `/validation/run` -> poll `/validation/jobs/{job_id}` -> retrieve report via `/validation/reports/{dataset_name}`.
2. Checkpoint evaluation: `/validation/checkpoint` -> poll `/validation/jobs/{job_id}` -> retrieve report via `/validation/checkpoint/reports/{checkpoint}`.

## 8. Runtime and Deployment

### 8.1 Local mode
- Typical launcher: `XREPORT/start_on_windows.bat`
- Uses local `.env` values and portable runtimes in `XREPORT/resources/runtimes` on Windows.

### 8.2 Cloud mode
- Docker Compose services:
  - `backend` (FastAPI/Uvicorn)
  - `frontend` (Nginx serving built frontend)
- Compose file: `docker-compose.yml`

## 9. Known Limitations

- No auth/RBAC on API routes.
- Uploaded dataset file state (`UploadState`) is in-memory and not persisted across backend restarts.
- Filesystem browse endpoint is Windows-drive oriented.
- `/training/stop` is a legacy compatibility endpoint; job cancellation via `/training/jobs/{job_id}` is the primary path.
- `vite.config.ts` defines websocket proxy entries, but backend route modules currently expose polling-based APIs only.
