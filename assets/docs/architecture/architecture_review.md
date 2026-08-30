# XREPORT Architecture Review

Last updated: 2026-08-30

This review records the implementation status of the architecture findings identified during the XREPORT review on the `develop` branch. It is a decision and remediation record, not a second source of endpoint or schema truth; the specialized architecture documents remain authoritative for current behavior.

## Summary

No P0 blocker remains. The reviewed P1 and P2 findings are implemented and
validated. The cleanup also removes internal fallback paths and competing
sources of truth: runtime layout, application configuration, checkpoint
identity, upload identity, job lifecycle, API schemas, and version authority
now each have one canonical owner.

## Findings And Status

| Priority | Finding | Status | Implementation |
| --- | --- | --- | --- |
| P1 | Existing databases had no startup schema compatibility check. | Resolved | Alembic startup checks the current revision, applies the checked-in migration stream under a migration lock, and fails clearly on missing, incompatible, or unsupported schema state; unversioned application tables are not silently adopted. |
| P1 | Generic `JobManager` contained inference-specific failure semantics. | Resolved | `JobManager` owns generic lifecycle state and fallback errors; feature services supply failure mapping when needed. |
| P1 | Inference history persistence failures could be swallowed after generation. | Resolved | `run_inference_job` raises a typed `persistence_failed` job error and leaves the job failed when history persistence fails. |
| P1 | Runtime paths, configuration, uploads, and checkpoint identity had competing fallback sources. | Resolved | `RuntimeLayout`, strict `.env`/JSON ownership, explicit upload IDs, and the database-owned `CheckpointRepository` are the only authorities; filesystem artifacts are verified rather than scanned for identity. |
| P1 | Feature-specific job routes and browser-side job snapshots duplicated lifecycle state. | Resolved | `/api/jobs` and `JobsApiService` provide the single status/cancellation surface; pages keep only transient UI state and derive results from `JobStatusResponse`. |
| P1 | Client-selected workflow parameters were silently defaulted by the backend. | Resolved | Request contracts require the selected dataset, upload confirmation, training, processing, validation, and evaluation parameters; the backend retains validation bounds without a second UX default policy. |
| P1 | `PreparationService` combined workflow coordination and dataset processing. | Resolved | `DatasetProcessingService` owns processing, splitting, tokenization, and processed-sample persistence behind an injected `DatasetRepository`. |
| P2 | Mutable `JobState` lived in the domain package. | Resolved | `JobState` and `JobExecutionError` now live in `services/jobs.py`; domain contains serializable job contracts only. |
| P2 | Service dependencies were partly hidden behind globals. | Resolved for reviewed composition paths | Preparation and inference factories compose job, repository, runtime, installation, and processing dependencies explicitly; job functions accept those dependencies as arguments. |
| P2 | One frontend `ApiService` crossed every backend feature. | Resolved | Shared transport is separated from inference, dataset, training, validation, and generic jobs API clients. Pages and components inject only the clients they use. |
| P2 | Architecture checks did not enforce the domain boundary strongly enough. | Resolved | The architecture test rejects domain imports of API, models, repositories, and services. |
| P3 | Documentation referenced a nonexistent Angular `hooks/*` layer. | Resolved | Architecture documentation now describes the actual services and polling boundaries. |

## Current Target Architecture

```mermaid
flowchart TD
    UI[Angular pages/components]
    UI --> Clients[Feature API clients]
    Clients --> Transport[Shared HTTP transport]
    Transport --> Routers[FastAPI routers]
    Routers --> Contracts[Domain contracts]
    Routers --> UseCases[Application services]
    UseCases --> Runtime[Job/runtime/model dependencies]
    UseCases --> Repos[Focused repositories]
    Repos --> ORM[SQLAlchemy schemas]
    ORM --> DB[(SQLite/PostgreSQL)]
```

The architecture keeps contracts stable while allowing service and provider implementations to evolve behind injected boundaries. Persistence ownership remains singular: dataset, validation, and inference histories are written by their respective repositories.

## Migration Policy

### Alembic migration history

Alembic is now the schema authority. The current head revision
`d62f3ab4e8c1` removes report-job state columns and constraints, then performs a
single controlled bootstrap of complete checkpoint artifacts into the database
registry. Startup and explicit initialization both upgrade to the single head,
with transaction/lock protection for concurrent processes; an existing
unversioned application schema is rejected rather than stamped implicitly.

### Broader service cleanup

Future service additions should preserve the explicit-composition rule. New jobs should use generic job errors plus feature-local mapping, and new persistence flows should surface repository failures rather than reporting successful completion.

## Validation Evidence

- Backend unit tests cover schema initialization, checkpoint registration and
  deletion guards, job failure semantics, cancellation, inference persistence
  failure, dataset processing, preparation scanning, explicit request
  parameters, and architecture imports.
- Angular lint, unit tests, generated OpenAPI types, and the production build
  cover the feature-client split and existing frontend behavior.
- The implementation was committed in incremental slices and pushed to `origin/develop`.

## Related Documentation

- `system_overview.md`: runtime topology and dependency direction.
- `execution_and_data_flow.md`: service composition and job/data sequences.
- `persistence.md`: schema compatibility and entity ownership.
- `backend_api.md`: current FastAPI endpoint contract and serving behavior.
