# Troubleshooting And Initialization

Last updated: 2026-10-20

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
- configured port already in use:
  - interactive Launch lists the owning PIDs, ports, and available process metadata before asking once for termination
  - answer `No` to cancel without stopping processes or starting services
  - non-interactive Launch fails closed; free the ports or rerun interactively
  - a failed termination, launcher/ancestor owner, or new owner appearing after approval aborts the launch without killing the new process
- frontend bundle is stale or missing:
  - run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action RebuildFrontend`
  - inspect the launcher timing and build-state reason; dependency manifest changes run `npm ci` before the rebuild
- jobs stay running too long:
  - poll the status endpoint and inspect backend logs under `<resource root>/logs`
- missing artifacts or checkpoints:
  - confirm write permissions and paths under the configured resource root
- first run is slow:
  - expected when dependencies and runtimes are being initialized
- model unavailable:
  - inspect `GET /api/inference/models` for provider and model status
  - Hugging Face model snapshots must already be cached at the exact revision declared in the typed catalogue at `app/server/configurations/inference_models.py`
  - Hugging Face requires a cached snapshot and an exact configured commit
- first Generate action is slow or reports an access error:
  - allow time for the selected public model to download, verify, and load
  - for gated models, accept the provider terms and configure `HF_TOKEN` or the standard local Hugging Face credential store
- startup validation failure:
  - inspect the database migration status and confirm the `application_settings`
    singleton row exists and passes its constraints
  - check write permissions under the configured resource root

### ML import deadlocks after startup

Lightweight endpoints such as dataset names/status, checkpoint listing, settings,
and the inference model catalogue must not initialize Keras, PyTorch, torchvision,
Transformers, or a model provider. The model package initializers are intentionally
empty, and service factories keep runtime imports inside tokenization, training,
checkpoint loading, validation, and inference execution paths.

If logs contain `_ModuleLock` deadlocks, `partially initialized module 'torch'`,
`torch.utils`, or Keras/PyTorch circular-import errors, restart from a clean
process and check that the request was not importing a concrete runtime module at
factory time. Use the clean-subprocess regression test and call the lightweight
endpoints before starting any ML job. A genuine runtime dependency failure should
remain attached to its job result and should be fixed in the execution path rather
than suppressed with retries or broad exception handling.

## Database Initialization

### SQLite

- Select SQLite in Settings → Database and set an absolute database file path. The default is `<resource root>/database.db`; packaged mode defaults to `%LOCALAPPDATA%\XREPORT\data\database.db` and never writes into the immutable installation/runtime directory.
- On later startups, existing data is not recreated, reset, or reseeded. Alembic checks the applied revision and upgrades to head under an exclusive SQLite transaction.
- A non-empty database without a recognized Alembic revision, or with a partial or modified schema, fails closed and is not repaired automatically.

### PostgreSQL

- Enter the host, port, database, username, password, and optional SSL settings in Settings → Database. A saved password is protected with Windows DPAPI for the current Windows account.
- PostgreSQL initialization creates the selected database when absent if the configured role has permission, then applies pending Alembic revisions under advisory locks.
- Use Test connection to distinguish network/authentication failures from a schema that needs initialization. Invalid, unavailable, unsupported, or permission-limited connections show a sanitized error; correct the settings or use an appropriately privileged account.

### Startup database recovery and switching

- A missing, corrupt, or unavailable database configuration opens the restricted Database screen. Normal API routes remain unavailable until a connection and schema are activated.
- Switching blocks new API requests while in-flight requests drain, and is unavailable while a background job is running. Each database keeps its own data and application settings; switching does not copy records.
- Legacy database keys in `.env` are migrated once when supported and removed. A legacy `DATABASE_URL` is unsupported; enter SQLite or PostgreSQL fields in recovery.
- Restart after changing process launch settings such as the resource root, host, or port. Switching a database from Settings does not require a restart.

### Schema compatibility failure

- Read the startup error for the missing table/column, incompatible unversioned schema, unknown Alembic revision, multiple heads, or migration failure.
- Preserve a copy of the database before any recovery action.
- If the database is disposable, recreate it through the documented initialization path.
- If the data is needed, preserve a backup and apply a reviewed supported migration. The current release intentionally fails fast on arbitrary schema changes.
