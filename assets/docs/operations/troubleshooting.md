# Troubleshooting And Initialization

Last updated: 2026-07-20

## Troubleshooting Quick Guide

- UI not reachable:
  - check `UI_HOST` and `UI_PORT` in `XREPORT/settings/.env`
  - verify the backend is running on `FASTAPI_HOST` and `FASTAPI_PORT`
- jobs stay running too long:
  - poll the status endpoint and inspect backend logs in `XREPORT/resources/logs`
- missing artifacts or checkpoints:
  - confirm write permissions and paths under `app/resources`
- first run is slow:
  - expected when dependencies and runtimes are being initialized
- model unavailable:
  - inspect `GET /api/inference/models` for provider and model status
  - Ollama models must already be installed and the local runtime must be reachable
  - Hugging Face requires a cached snapshot and an exact configured commit
- startup validation failure:
  - verify `settings/configurations.json` exists
  - check write permissions under `app/resources`
  - set `XREPORT_TAURI_MODE=false` unless a built frontend is present

## Database Initialization

### SQLite Mode

- When `XREPORT_DB_BACKEND=sqlite`, the backend initializes `XREPORT/resources/database.db` automatically on first startup if the file does not exist.
- On later startups, the schema is validated. For the inference-first branch, stop the app and delete `app/resources/database.db` if startup reports legacy inference columns; the database is recreated on restart.

### PostgreSQL Mode

- When `XREPORT_DB_BACKEND=postgresql`, PostgreSQL initialization uses the database values from `XREPORT/settings/.env`.
- Run `start_on_windows.ps1`, choose `3. Initialize database`, and execute `app/scripts/initialize_database.py`.
- The same script also works for SQLite mode, but it is normally unnecessary because first-run SQLite initialization is automatic.
- Use a disposable PostgreSQL database for this feature branch. If legacy inference columns are reported, drop and recreate that feature database before running initialization; `create_all` is not a migration mechanism.
