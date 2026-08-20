# XREPORT Architecture Review

Last updated: 2026-08-20

This review records the implementation status of the architecture findings identified during the XREPORT review on the `develop` branch. It is a decision and remediation record, not a second source of endpoint or schema truth; the specialized architecture documents remain authoritative for current behavior.

## Summary

No P0 blocker remains. The P1 findings were implemented and validated. The main P2 structural findings were also addressed: mutable job state is outside the domain package, service composition is explicit for preparation and inference, and the frontend API gateway is split by feature. Alembic remains intentionally deferred until the project adopts a migration policy for long-lived databases.

## Findings And Status

| Priority | Finding | Status | Implementation |
| --- | --- | --- | --- |
| P1 | Existing databases had no startup schema compatibility check. | Resolved | `schema_metadata` records version 1; startup checks required tables and columns and fails clearly on incompatible or unsupported schemas. Structurally compatible unversioned databases are stamped. |
| P1 | Generic `JobManager` contained inference-specific failure semantics. | Resolved | `JobManager` owns generic lifecycle state and fallback errors; feature services supply failure mapping when needed. |
| P1 | Inference history persistence failures could be swallowed after generation. | Resolved | `run_inference_job` raises a typed `persistence_failed` job error and leaves the job failed when history persistence fails. |
| P1 | `PreparationService` combined workflow coordination and dataset processing. | Resolved | `DatasetProcessingService` owns processing, splitting, tokenization, and processed-sample persistence behind an injected `DatasetRepository`. |
| P2 | Mutable `JobState` lived in the domain package. | Resolved | `JobState` and `JobExecutionError` now live in `services/jobs.py`; domain contains serializable job contracts only. |
| P2 | Service dependencies were partly hidden behind globals. | Resolved for reviewed composition paths | Preparation and inference factories compose job, repository, runtime, installation, and processing dependencies explicitly; job functions accept those dependencies as arguments. |
| P2 | One frontend `ApiService` crossed every backend feature. | Resolved | Shared transport is separated from inference, dataset, training, and validation API clients. Pages and components inject only the clients they use. |
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

## Deferred Work

### Alembic migration history

The current release adds a version marker and compatibility guard, but it does not invent an upgrade path for arbitrary existing schemas. Alembic should be introduced when the project has a migration ownership and deployment policy, including supported upgrade/downgrade expectations, PostgreSQL rollout rules, and a test fixture strategy. Until then, incompatible schemas fail fast with recreate-or-migrate guidance.

### Broader service cleanup

Future service additions should preserve the explicit-composition rule. New jobs should use generic job errors plus feature-local mapping, and new persistence flows should surface repository failures rather than reporting successful completion.

## Validation Evidence

- Backend focused tests cover schema initialization, job failure semantics, cancellation, inference persistence failure, dataset processing, preparation scanning, and architecture imports.
- Angular lint, unit tests, and production build cover the feature-client split and existing frontend behavior.
- The implementation was committed in incremental slices and pushed to `origin/develop`.

## Related Documentation

- `system_overview.md`: runtime topology and dependency direction.
- `execution_and_data_flow.md`: service composition and job/data sequences.
- `persistence.md`: schema compatibility and entity ownership.
- `backend_api.md`: current FastAPI endpoint contract and serving behavior.
