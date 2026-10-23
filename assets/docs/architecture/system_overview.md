# XREPORT System Overview

Last updated: 2026-10-21

XREPORT is a local-first client/server system for radiological report generation, dataset preparation, training, validation, and model lifecycle workflows.

## Runtime Topology

- Frontend: standalone Angular 22 + TypeScript in `app/client`.
- Backend: FastAPI in `app/server`.
- API contracts: transport-neutral Pydantic models in `app/server/domain`,
  mirrored in the checked-in per-application snapshots
  `app/shared/openapi.xreport.json` and `openapi.studio.json` and generated
  Angular types. See [applications and checkpoint exchange](applications_and_checkpoint_exchange.md)
  for the XREPORT / XREPORT Studio split.
- Backend orchestration: services in `app/server/services`.
- Persistence: SQLAlchemy repositories and schemas in `app/server/repositories`, with SQLite by default and PostgreSQL optional.
- Learning runtime: model, provider, and inference implementation details in `app/server/models`.
- Long-running execution: start, poll, and cancel jobs through the generic
  `app/server/services/jobs.py` lifecycle and `/api/jobs` resource.
- Startup validation: Alembic database creation/upgrade and required resource-directory checks before the API serves requests.

The Angular application is a static SPA and uses relative `/api` requests. In
normal Windows source Launch, FastAPI serves the built Angular files and API
from one origin. The launcher waits for `/api/health`, opens the FastAPI URL,
and remains the lifecycle owner until the user stops the app. Its opt-in
`XREPORT_SERVE_CLIENT` flag leaves default source-mode root behavior unchanged.
Angular CLI remains the frontend runtime for development workflows.

### Packaged Windows topology

The Tauri webview loads Angular from the bundled `ui/` assets declared by
`frontendDist`. Rust owns runtime extraction, the common per-user mutex,
dynamic-port readiness/health polling, the Windows Job Object, navigation
policy, and sidecar shutdown. FastAPI serves the API and retains its existing
packaged security and SPA routes; Tauri does not use backend static hosting to
load its frontend. Immutable runtime files are separated from mutable
`%LOCALAPPDATA%\XREPORT\data`.

`LaunchDesktopDev` runs Tauri in the launcher console and supervises its source
backend and built-bundle development server, keeping their logs under the
resource root. `npm preview` is for bundle validation and Tauri development,
not production hosting.

## Dependency Direction

```mermaid
flowchart LR
    Browser --> Pages[Angular pages and components]
    Pages --> FeatureClients[Feature API clients]
    FeatureClients --> Transport[ApiRequestService]
    Transport --> FastAPI[FastAPI routers]
    FastAPI --> Contracts[Domain contracts]
    FastAPI --> Services[Application services]
    Services --> Jobs[JobManager]
    Services --> Processing[DatasetProcessingService]
    Services --> Repositories[Domain repositories]
    Services --> Learning[Model and inference runtime]
    Repositories --> Schemas[SQLAlchemy schemas]
    Schemas --> Database[(SQLite or PostgreSQL)]
```

The intended dependency rule is inward and explicit: transport adapts requests to contracts, services orchestrate use cases, repositories own persistence, and learning modules own model execution. Domain contracts do not import services, repositories, API routers, or mutable job runtime state.

## Implementation-Relevant Repository Structure

```text
.
├─ README.md
├─ start_on_windows.ps1
├─ runtimes/
│  ├─ python/
│  ├─ uv/
│  └─ nodejs/
├─ assets/
│  └─ docs/
├─ settings/
│  └─ .env.example
├─ data/
└─ app/
   ├─ scripts/
   │  └─ initialize_database.py
   ├─ server/
   │  ├─ pyproject.toml
   │  ├─ api/
   │  ├─ common/
   │  ├─ domain/
   │  ├─ configurations/
   │  │  └─ inference_models.py
   │  ├─ models/
   │  ├─ services/
   │  └─ repositories/
   ├─ client/
   │  ├─ package.json
   │  ├─ angular.json
   │  └─ src/
   └─ tests/
      └─ run_tests.bat
```

## Backend Boundaries

### API and domain

`app/server/api` owns HTTP parsing, router registration, response models, and service-error-to-HTTP translation. `app/server/domain` contains stable request and response contracts only. `JobState` and `JobExecutionError` are service-runtime concerns, not domain contracts.

### Services

`app/server/services` contains use-case orchestration and explicit dependency composition. `DatasetProcessingService` owns tokenization, split calculation, processed-row persistence, and processing metadata. `PreparationService` owns preparation workflow state and delegates processing. `InferenceService` receives its runtime, installation, repository, and job dependencies through its factory.

### Repositories and learning runtime

`app/server/repositories` contains SQLAlchemy schemas plus focused dataset,
checkpoint, validation, and inference persistence boundaries. Checkpoint
identity is database-owned; the filesystem stores only the registered model
artifacts. `app/server/models` contains model-training and inference
implementation details; it is not the ORM model layer. ORM metadata lives in
`repositories/schemas`; Alembic revisions and the migration environment live in
`server/migrations`.

## Frontend Boundaries

Route pages and reusable components inject feature-specific clients:

- `services/inference-api.service.ts`: model catalogue, maintenance, and generation.
- `services/dataset-api.service.ts`: upload, loading, browsing, processing, and image inspection.
- `services/training-api.service.ts`: checkpoints and training.
- `services/validation-api.service.ts`: dataset validation and checkpoint evaluation.
- `services/jobs-api.service.ts`: generic job listing, status, and cancellation.
- `services/api-request.service.ts`: shared HTTP execution and error-envelope formatting.
- `services/job-polling.service.ts`: shared polling behavior over the generic job client.

This keeps page dependencies aligned with the feature they operate on and avoids a single cross-domain API gateway.

## Entry Points

- Backend API entrypoint: `app/server/app.py`.
- Frontend web entrypoints: `app/client/src/main.ts` (XREPORT) and `app/client/src/studio/main.ts` (Studio).
- Frontend route composition: `app/client/src/app/app.routes.ts` (XREPORT) and `app/client/src/studio/studio.routes.ts` (Studio).
- OpenAPI source snapshot and generated client types: `app/shared/openapi.{xreport,studio}.json` and `app/client/src/app/types/generated/{xreport,studio}/types.gen.ts`.
- Release version authority: `app/server/pyproject.toml`; client, desktop,
  Tauri, backend metadata, and OpenAPI copies are validated against it.
- Local launcher and maintenance menu on Windows: `start_on_windows.ps1`.
- Explicit database initialization: `app/scripts/initialize_database.py` and launcher option 4.
