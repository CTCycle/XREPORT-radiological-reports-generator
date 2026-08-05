# XREPORT Execution And Data Flow

Last updated: 2026-08-05

## Layer Responsibilities

### Endpoint Layer

Location: `app/server/api`

- Parses transport concerns such as multipart files plus path, query, and body parameters.
- Converts HTTP interactions into service calls.
- Applies response models and status codes.
- Maps typed service failures to the existing HTTP status and `{"detail": ...}` envelope through one registered exception handler.

### Domain Layer

Location: `app/server/domain`

- Defines transport-neutral request and response models for inference, jobs, training, and validation.
- Keeps endpoint contracts separate from service orchestration and provider implementations.

### Service Layer

Location: `app/server/services`

- Contains orchestration and business rules.
- Starts and monitors long-running jobs.
- Maps repository results into API and domain responses.
- Owns training-process orchestration in `training_worker.py` and inference-catalog orchestration in `inference_catalog.py`.
- Raises typed `ServiceError` subclasses for expected failures; the API layer performs the HTTP translation.
- Loads serialized checkpoint artifacts before passing models and metadata to inference providers.
- Owns startup preparation through `startup_validation.py`, coordinating database preparation and resource validation before the application serves requests.

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
- Inference providers sit behind the catalog-selected `model_ref`. External models use one manifest-driven embedded Hugging Face Transformers runtime; custom XREPORT checkpoints retain their dedicated Keras/BEiT path. A shared runtime lock serializes model residency across both paths and unloads Hugging Face state before an XREPORT checkpoint or another Hugging Face key is loaded.
- The catalog reads `settings/inference_models.json` and reports installation state. First-use download is owned by the background inference job, not by catalog reads.
- `inference_runtime.py` is the single coordinator for XREPORT and Hugging Face generation. It owns the justified single-user lock, one-model residency, installation coordination, and typed provider results.
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
- Inference jobs retain uploaded images at the service boundary, publish per-request progress/results through the job manager, and persist final metadata/reports through `InferenceRepository`. Result payloads keep the exact `reports` mapping and add declared `display_sections` plus provenance containing provider, model/ref/revision, loaders, adapter, prompt/generation profiles, clinical context, image dimensions, and processed tensor dimensions.
- Uploaded image bytes are linked to the job by an internal request ID and removed when the job completes, is cancelled, or fails to start.
- Database access is synchronous through SQLAlchemy engines and sessions. No async database driver is part of the current implementation.

## Hugging Face generation contract

Manifest entries are strict, SHA-pinned contracts. Required files and at least one non-empty weight-file alternative are downloaded into project-local staging, then verified before activation. Remote code is rejected unless the manifest explicitly approves it. Generation has deterministic profile limits, an `INFERENCE_MODEL_TIMEOUT` deadline, and cooperative job cancellation; cancelled or timed-out partial output is not persisted. EXIF orientation and RGB conversion occur before processor calls, while integer token dtypes are preserved when floating image tensors are moved to the model device and dtype.

Every Hugging Face generation uses a complete manifest and a verified project-local snapshot. Provider-only manifests, unverified cache discovery, and global-cache fallback loading are not supported.

## Architectural Constraints

- The system is local-first and filesystem-aware. Local path browsing is part of the supported workflow.
- No authentication or authorization layer is implemented in the current API surface.
- Job progress is polling-based. No production WebSocket API is currently exposed by backend routes.
