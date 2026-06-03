# XREPORT Persistence

Last updated: 2026-06-03

## Database Backend Selection

From `XREPORT/settings/configurations.json`:

- `database.embedded_database=true`: SQLite using `XREPORT/resources/database.db`
- `database.embedded_database=false`: PostgreSQL using configured engine, host, port, database name, user, password, and SSL settings

## Initialization Behavior

- Backend startup performs database initialization before serving requests.
- SQLite mode ensures schema creation against the embedded database file.
- PostgreSQL mode executes database and schema initialization from configured connection settings.
- Additional startup validation ensures required resource directories exist.
- In Tauri mode, startup also validates that a built frontend bundle is available before serving requests.

## Persisted Domains

Core persisted entities include:

- datasets and dataset records
- processing runs and training samples
- validation runs plus text and image aggregates plus pixel distributions
- checkpoints and checkpoint evaluations
- inference runs and generated reports

## Non-Database Artifacts

- checkpoints and model artifacts under `XREPORT/resources/checkpoints` and `XREPORT/resources/models`
- logs under `XREPORT/resources/logs`
- templates under `XREPORT/resources/templates`
- tokenizer resources under `XREPORT/resources/tokenizers`
