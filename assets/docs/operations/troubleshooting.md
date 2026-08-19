# Troubleshooting And Initialization

Last updated: 2026-08-19

## Troubleshooting Quick Guide

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

- When `EMBEDDED_DATABASE=true`, the backend initializes `<resource root>/database.db` automatically on first startup if the file does not exist. The resource root defaults to `app/resources` and can be changed with `XREPORT_RESOURCES_DIR`.
- On later startups, only the file existence check is performed. Existing data is not recreated, reset, reseeded, or schema-validated.
- The launcher option `4` can be used to manually trigger the same idempotent SQLite initialization.

### PostgreSQL Mode

- When `EMBEDDED_DATABASE=false`, normal startup only checks the configured PostgreSQL connection and does not create or initialize anything.
- Run `start_on_windows.ps1`, choose `4. Initialize database`, and execute the explicit initialization before launching the application.
- The same command also works for SQLite mode and is safe when the SQLite file already exists.
- Invalid, unavailable, or not-yet-initialized PostgreSQL connections fail startup with a database startup error; correct the connection settings or run the explicit initialization command.
