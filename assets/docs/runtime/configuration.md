# Runtime Configuration

Last updated: 2026-06-05

## Shared Configuration Sources

- Environment overrides: `XREPORT/settings/.env`
- Static configuration: `XREPORT/settings/configurations.json`

## Key Environment Variables

- `FASTAPI_HOST`
- `FASTAPI_PORT`
- `UI_HOST`
- `UI_PORT`
- `VITE_API_BASE_URL`
- `RELOAD`
- `OPTIONAL_DEPENDENCIES`
- `MPLBACKEND`
- `KERAS_BACKEND`
- `XREPORT_DB_EMBEDDED`
- `XREPORT_DATABASE_URL`
- `XREPORT_DB_ENGINE`
- `XREPORT_DB_HOST`
- `XREPORT_DB_PORT`
- `XREPORT_DB_NAME`
- `XREPORT_DB_USERNAME`
- `XREPORT_DB_PASSWORD`
- `XREPORT_DB_SSL`
- `XREPORT_DB_SSL_CA`
- `XREPORT_DB_CONNECT_TIMEOUT`
- `XREPORT_DB_INSERT_BATCH_SIZE`

Expected value note:

- `VITE_API_BASE_URL` should remain `/api` for the proxied local flow.

## Database Mode Switch

From `XREPORT/settings/.env`:

- `XREPORT_DB_EMBEDDED=true` selects SQLite
- `XREPORT_DB_EMBEDDED=false` selects PostgreSQL

Initialization differences:

- SQLite ensures schema initialization at backend startup.
- PostgreSQL performs database and schema initialization during backend startup using `.env` connection settings.

## Interoperability

- Frontend calls backend routes through `/api`.
- Vite dev and preview proxy `/api` to `http://FASTAPI_HOST:FASTAPI_PORT`.
- Tauri desktop starts the backend locally, waits for TCP readiness, then redirects the desktop window to the backend root URL.
- Backend serves packaged SPA assets from `app/client/dist` when a frontend build is available.
- In Tauri mode, backend startup also validates that the packaged frontend build is present before serving requests.
