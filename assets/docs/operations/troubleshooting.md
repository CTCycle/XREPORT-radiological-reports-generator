# Troubleshooting And Initialization

Last updated: 2026-08-21

## Troubleshooting Quick Guide

### Desktop startup failures

If the desktop splash changes to the startup error page, inspect
`%LOCALAPPDATA%\XREPORT\data\logs\desktop-shell.log` and the timestamped
backend log. The shell rejects stale or mismatched runtime manifests, waits for
the readiness contract plus authenticated `/api/health`, and reports an early
backend exit instead of leaving an orphan process. It sends a bounded graceful
shutdown request first; the Windows Job Object terminates descendants if the
timeout expires.

The packaged backend chooses a free loopback port, so a process using 5003 does
not prevent startup. Only one CPU/CUDA/portable/installed XREPORT desktop
instance is allowed per Windows user. MSI uninstall preserves user data;
remove `%LOCALAPPDATA%\XREPORT\data` manually only for a complete reset.
The CPU and CUDA products intentionally share that data directory while using
variant-specific immutable runtime archives.

- UI not reachable:
  - check `UI_HOST` and `UI_PORT` in `settings/.env`
  - verify the backend is running on `FASTAPI_HOST` and `FASTAPI_PORT`
- jobs stay running too long:
  - poll the status endpoint and inspect backend logs under `<resource root>/logs`
- missing artifacts or checkpoints:
  - confirm write permissions and paths under the configured resource root
- first run is slow:
  - expected when dependencies and runtimes are being initialized
- model unavailable:
  - inspect `GET /api/inference/models` for provider and model status
  - Hugging Face model snapshots must already be cached at the exact revision declared in `settings/inference_models.json`
  - Hugging Face requires a cached snapshot and an exact configured commit
- first Generate action is slow or reports an access error:
  - allow time for the selected public model to download, verify, and load
  - for gated models, accept the provider terms and configure `HF_TOKEN` or the standard local Hugging Face credential store
- startup validation failure:
  - verify `settings/configurations.json` exists
  - check write permissions under the configured resource root

## Database Initialization

### SQLite Mode

- When `EMBEDDED_DATABASE=true`, source mode initializes `<resource root>/database.db` automatically on first startup if the file does not exist. Packaged mode initializes `%LOCALAPPDATA%\XREPORT\data\database.db`; it never writes a database into the immutable installation/runtime directory.
- On later startups, existing data is not recreated, reset, or reseeded. Alembic checks the applied revision and upgrades to head under an exclusive SQLite transaction.
- A non-empty database without `alembic_version` is stamped only after exact v1 schema validation. Partial or modified schemas fail and are not repaired automatically.
- The launcher option `4` can be used to manually trigger the same idempotent Alembic initialization.

### PostgreSQL Mode

- When `EMBEDDED_DATABASE=false`, startup and option `4` create the configured PostgreSQL database when absent (subject to permissions), then apply pending Alembic revisions under advisory locks.
- The same command also works for SQLite mode and is safe when the database already exists.
- Invalid, unavailable, unsupported, or permission-limited PostgreSQL connections fail with a sanitized database migration error; correct the connection settings or use an administrative initialization account.

### Schema compatibility failure

- Read the startup error for the missing table/column, incompatible unversioned schema, unknown Alembic revision, multiple heads, or migration failure.
- Preserve a copy of the database before any recovery action.
- If the database is disposable, recreate it through the documented initialization path.
- If the data is needed, preserve a backup and apply a reviewed supported migration. The current release intentionally fails fast on arbitrary schema changes.
