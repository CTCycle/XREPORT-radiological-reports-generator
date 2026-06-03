# Runtime Configuration

Last updated: 2026-06-03

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

Expected value note:

- `VITE_API_BASE_URL` should remain `/api` for the proxied local flow.

## Database Mode Switch

From `configurations.json`:

- `database.embedded_database=true` selects SQLite
- `database.embedded_database=false` selects PostgreSQL

Initialization differences:

- SQLite ensures schema initialization at backend startup.
- PostgreSQL performs database and schema initialization during backend startup using configured connection settings.

## Interoperability

- Frontend calls backend routes through `/api`.
- Vite dev and preview proxy `/api` to `http://FASTAPI_HOST:FASTAPI_PORT`.
- Tauri desktop starts the backend locally, waits for TCP readiness, then redirects the desktop window to the backend root URL.
- Backend serves packaged SPA assets from `app/client/dist` when a frontend build is available.
- In Tauri mode, backend startup also validates that the packaged frontend build is present before serving requests.
