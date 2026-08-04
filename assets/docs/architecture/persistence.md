# XREPORT Persistence

Last updated: 2026-08-03

## Database Backend Selection

From `XREPORT/settings/.env`:

- `EMBEDDED_DATABASE=true`: SQLite using `XREPORT/resources/database.db`
- `EMBEDDED_DATABASE=false`: PostgreSQL using configured engine, host, port, database name, user, password, and SSL settings

## Initialization Behavior

- Backend startup uses one shared lifecycle entrypoint before serving requests.
- If `settings/.env` is missing, it is copied from `settings/.env.example`; an
  existing environment file is never overwritten.
- SQLite startup checks only whether `app/resources/database.db` exists. A
  missing file is initialized once; an existing file is not recreated, reset,
  reseeded, or schema-validated.
- PostgreSQL startup performs only a connection check against the configured
  database. Database and schema creation are explicit through option `3` in
  `start_on_windows.ps1`.
- The repository has no applicable database seed routine; initialization only
  creates the existing schema.
- Additional startup validation ensures required resource directories exist.

The current schema records provider, model reference and revision, generation
profile/configuration, clinical context, request ID, lifecycle status,
execution timestamp, and execution duration. Generated-report persistence is
owned by `InferenceRepository`.

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
