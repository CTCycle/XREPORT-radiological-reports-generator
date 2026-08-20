# Troubleshooting And Initialization

Last updated: 2026-08-20

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
- startup validation failure:
  - verify `settings/configurations.json` exists
  - check write permissions under the configured resource root

## Database Initialization

### SQLite Mode

- When `EMBEDDED_DATABASE=true`, source mode initializes `<resource root>/database.db` automatically on first startup if the file does not exist. Packaged mode initializes `%LOCALAPPDATA%\XREPORT\data\database.db`; it never writes a database into the immutable installation/runtime directory.
- On later startups, existing data is not recreated, reset, or reseeded. The required tables and columns are checked against schema version 1. A structurally compatible older database without `schema_metadata` is stamped with the current marker; an incompatible schema, missing marker row, or unsupported version stops startup.
- The launcher option `4` can be used to manually trigger the same idempotent SQLite initialization.

### PostgreSQL Mode

- When `EMBEDDED_DATABASE=false`, normal startup checks the configured PostgreSQL connection and verifies schema compatibility; it does not create or migrate the schema.
- Run `start_on_windows.ps1`, choose `4. Initialize database`, and execute the explicit initialization before launching the application.
- The same command also works for SQLite mode and is safe when the SQLite file already exists.
- Invalid, unavailable, or not-yet-initialized PostgreSQL connections fail startup with a database startup error; correct the connection settings or run the explicit initialization command.

### Schema compatibility failure

- Read the startup error for the missing table, missing column, missing `xreport` marker, or unsupported schema version.
- Preserve a copy of the database before any recovery action.
- If the database is disposable, recreate it through the documented initialization path.
- If the data is needed, apply a supported migration once one exists; the current release does not transform arbitrary schema changes and intentionally fails fast.
