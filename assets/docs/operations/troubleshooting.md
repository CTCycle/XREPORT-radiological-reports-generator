# Troubleshooting And Initialization

Last updated: 2026-07-11

## Troubleshooting Quick Guide

- UI not reachable:
  - check `UI_HOST` and `UI_PORT` in `XREPORT/settings/.env`
  - verify the backend is running on `FASTAPI_HOST` and `FASTAPI_PORT`
- jobs stay running too long:
  - poll the status endpoint and inspect backend logs in `XREPORT/resources/logs`
- missing artifacts or checkpoints:
  - confirm write permissions and paths under `XREPORT/resources`
- first run is slow:
  - expected when dependencies and runtimes are being initialized

## Database Initialization

### SQLite Mode

- When `XREPORT_DB_EMBEDDED=true`, the backend initializes `XREPORT/resources/database.db` automatically on first startup if the file does not exist.
- On later startups, initialization is skipped when the file is already present.

### PostgreSQL Mode

- When `XREPORT_DB_EMBEDDED=false`, PostgreSQL initialization uses the database values from `XREPORT/settings/.env`.
- Run `start_on_windows.ps1`, choose `3. Initialize database`, and execute `app/scripts/initialize_database.py`.
- The same script also works for SQLite mode, but it is normally unnecessary because first-run SQLite initialization is automatic.
