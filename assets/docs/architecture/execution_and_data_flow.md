# XREPORT Execution And Data Flow

Last updated: 2026-07-14

## Layer Responsibilities

### Endpoint Layer

Location: `XREPORT/server/api`

- Parses transport concerns such as multipart files plus path, query, and body parameters.
- Converts HTTP interactions into service calls.
- Applies response models and status codes.

### Service Layer

Location: `XREPORT/server/services`

- Contains orchestration and business rules.
- Starts and monitors long-running jobs.
- Maps repository results into API and domain responses.

### Repository Layer

Location: `XREPORT/server/repositories`

- `database/*`: backend engine creation and database initialization
- `schemas/*`: SQLAlchemy table definitions
- `queries/*`: data access adapters
- `serialization/dataset.py`: dataset, processing, and training-data persistence
- `serialization/validation.py`: validation aggregate persistence boundary
- `serialization/inference.py`: inference and checkpoint-history persistence boundary
- `serialization/support.py`: shared JSON and UTC normalization

### Learning Layer

Location: `XREPORT/server/models`

- Holds model training and inference implementation details.
- Includes trainer, scheduler, dataloader, callback, and generator logic.

### Frontend Layer

Location: `XREPORT/client/src`

- `pages/*`: route-level workflows
- `components/*`: reusable UI building blocks
- `services/*`: backend API integration and polling
- `hooks/*`: reusable async and job-state patterns

## Async Versus Sync Behavior

- Most backend operations are synchronous request handlers that delegate CPU-heavy or long-running work to background jobs through `threading.Thread` via the job manager.
- Async handlers are used where the call path needs async I/O.
- Current async-sensitive cases include multipart file reads for upload and inference plus async validation endpoints that delegate to async service methods.
- Long-running compute is not executed directly inside request scope.
- Training uses managed job execution and a process worker pipeline.
- Preparation, validation, and inference heavy tasks follow start, poll, and cancel flows.
- Database access is synchronous through SQLAlchemy engines and sessions. No async database driver is part of the current implementation.

## Architectural Constraints

- The system is local-first and filesystem-aware. Local path browsing is part of the supported workflow.
- No authentication or authorization layer is implemented in the current API surface.
- Job progress is polling-based. No production WebSocket API is currently exposed by backend routes.
