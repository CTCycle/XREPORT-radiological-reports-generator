# XREPORT Execution And Data Flow

Last updated: 2026-07-18

## Layer Responsibilities

### Endpoint Layer

Location: `app/server/api`

- Parses transport concerns such as multipart files plus path, query, and body parameters.
- Converts HTTP interactions into service calls.
- Applies response models and status codes.
- Maps typed service failures to the existing HTTP status and `{"detail": ...}` envelope through one registered exception handler.

### Service Layer

Location: `app/server/services`

- Contains orchestration and business rules.
- Starts and monitors long-running jobs.
- Maps repository results into API and domain responses.
- Owns training-process orchestration in `training_worker.py` and inference-catalog orchestration in `inference_catalog.py`.
- Loads serialized checkpoint artifacts before passing models and metadata to inference providers.

### Repository Layer

Location: `app/server/repositories`

- `database/*`: backend engine creation and database initialization
- `schemas/*`: SQLAlchemy table definitions
- `queries/*`: data access adapters
- `serialization/dataset.py`: dataset, processing, and training-data persistence
- `serialization/validation.py`: validation aggregates and checkpoint-evaluation persistence
- `serialization/inference.py`: inference-run and generated-report persistence boundary
- `serialization/support.py`: shared database/session operations, entity lookup, and JSON/date normalization used by the three independent repositories

### Learning Layer

Location: `app/server/models`

- Holds model training and inference implementation details.
- Includes preprocessing/tokenization, trainer, scheduler, dataloader, callback, and generator logic.
- Inference providers sit behind the catalog-selected `model_ref`. Ollama uses its loopback API, and MedGemma loads only a pinned local snapshot.
- Model modules do not import services or repositories; required artifacts and cancellation state are injected by services.

### Frontend Layer

Location: `app/client/src`

- `pages/*`: route-level workflows
- `components/*`: reusable UI building blocks
- `services/*`: backend API integration and polling
- `hooks/*`: reusable async and job-state patterns

## Async Versus Sync Behavior

- Most backend operations are synchronous request handlers that delegate CPU-heavy or long-running work to background jobs through `threading.Thread` via the job manager.
- Async handlers are used where the call path needs async I/O.
- Current async-sensitive cases include multipart file reads for upload and inference plus async validation endpoints that delegate to async service methods.
- Long-running compute is not executed directly inside request scope.
- Training uses the service-owned managed process worker pipeline.
- Preparation, validation, and inference heavy tasks follow start, poll, and cancel flows.
- Inference jobs retain uploaded images at the service boundary, publish per-request progress/results through the job manager, and persist final metadata/reports through `InferenceRepository`.
- Database access is synchronous through SQLAlchemy engines and sessions. No async database driver is part of the current implementation.

## Architectural Constraints

- The system is local-first and filesystem-aware. Local path browsing is part of the supported workflow.
- No authentication or authorization layer is implemented in the current API surface.
- Job progress is polling-based. No production WebSocket API is currently exposed by backend routes.
