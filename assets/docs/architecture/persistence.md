# XREPORT Persistence

Last updated: 2026-07-18

## Database Backend Selection

From `XREPORT/settings/.env`:

- `XREPORT_DB_BACKEND=sqlite`: SQLite using `XREPORT/resources/database.db`
- `XREPORT_DB_BACKEND=postgresql`: PostgreSQL using configured engine, host, port, database name, user, password, and SSL settings

## Initialization Behavior

- Backend startup performs database initialization before serving requests.
- SQLite mode ensures schema creation against the local database file.
- PostgreSQL mode executes database and schema initialization from `.env` connection settings.
- Additional startup validation ensures required resource directories exist.

## Inference-First Branch Recreation Policy

This feature branch intentionally uses a clean database recreation instead of legacy inference migrations. SQLAlchemy `create_all` creates missing tables but does not migrate existing columns. Startup validates the `inference_runs` shape and fails with a recreation instruction when it detects the legacy schema.

- SQLite: stop XREPORT, delete `app/resources/database.db`, then restart or run database initialization.
- PostgreSQL: use a disposable feature-branch database and drop/recreate that database before initialization. Do not point this branch at a database whose data must be retained.

The inference-first schema makes checkpoint linkage nullable and records provider, model reference and revision, generation profile/configuration, clinical context, request ID, lifecycle status, execution timestamp, and execution duration. Generated-report persistence is owned by `InferenceRepository`.

`DatasetRepository`, `ValidationRepository`, and `InferenceRepository` are independent domain repositories. They share only focused `RepositorySupport` primitives for database injection, sessions, generic table operations, date/JSON normalization, and dataset/checkpoint lookup; none inherits another domain repository.

SQLite connections enable foreign-key enforcement, WAL journaling, normal
synchronous mode, and a 30-second busy timeout. All dataframe persistence
batches now run inside one transaction and roll back together on failure.

## Persisted Domains

Core persisted entities include:

- datasets and dataset records
- processing runs and training samples
- validation runs plus text and image aggregates plus pixel distributions
- checkpoints and checkpoint evaluations
- inference runs and generated reports

Persistence ownership is singular: dataset versioning, processing, and training samples belong to `DatasetRepository`; validation runs, validation aggregates, and checkpoint evaluations belong to `ValidationRepository`; inference runs and generated reports belong to `InferenceRepository`.

Dataset and record logical identities use normalized Unicode NFKC, trimmed,
case-folded keys. Each source import is an immutable `dataset_versions` snapshot:
identical content reuses its version, while changed content creates a new version.
Readers resolve the latest version, so removed images do not leak into current
processing while historical runs can retain their original record IDs.

Validation aggregate values are stored only on `validation_runs` and returned in one
primary query. Inference and checkpoint-evaluation history expose bounded
pagination with stable timestamp/ID ordering. Checkpoint history is retained by
restricting database deletion while evaluations or inference runs reference it.

## Non-Database Artifacts

- checkpoints and model artifacts under `XREPORT/resources/checkpoints` and `XREPORT/resources/models`
- logs under `XREPORT/resources/logs`
- templates under `XREPORT/resources/templates`
- tokenizer resources under `XREPORT/resources/tokenizers`
