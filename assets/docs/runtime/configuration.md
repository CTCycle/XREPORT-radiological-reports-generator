# Runtime Configuration

Last updated: 2026-07-15

## Shared Configuration Sources

- Environment overrides: `settings/.env`
- Tracked environment template: `settings/.env.example`
- Static configuration: `settings/configurations.json`

## Key Environment Variables

- `FASTAPI_HOST`
- `FASTAPI_PORT`
- `UI_HOST`
- `UI_PORT`
- `VITE_API_BASE_URL`
- `RELOAD`
- `BACKEND_VISIBLE`
- `OPTIONAL_DEPENDENCIES`
- `MPLBACKEND`
- `KERAS_BACKEND`
- `XREPORT_DB_BACKEND` (`sqlite` or `postgresql`)
- `XREPORT_DATABASE_URL`
- `XREPORT_DB_ENGINE` (`postgres`, `postgresql`, `postgresql+psycopg`, or
  `postgresql+psycopg2` when external mode is selected)
- `XREPORT_DB_HOST`
- `XREPORT_DB_PORT`
- `XREPORT_DB_NAME`
- `XREPORT_DB_USERNAME`
- `XREPORT_DB_PASSWORD`
- `XREPORT_DB_SSL`
- `XREPORT_DB_SSL_CA`
- `XREPORT_DB_CONNECT_TIMEOUT`
- `XREPORT_DB_INSERT_BATCH_SIZE`
- `XREPORT_OLLAMA_BASE_URL` (loopback local Ollama endpoint)
- `XREPORT_OLLAMA_KEEP_ALIVE` (model residency passed to `/api/chat`, default `5m`)
- `XREPORT_INFERENCE_MODEL_TIMEOUT` (generation read timeout in seconds)

`VITE_API_BASE_URL` should remain `/api` for the proxied local flow. Set `BACKEND_VISIBLE=true` to open backend logs in a dedicated terminal; the default keeps the backend window hidden.

## Database Mode Switch

- `XREPORT_DB_BACKEND=sqlite` selects SQLite.
- `XREPORT_DB_BACKEND=postgresql` selects PostgreSQL and requires the external
  connection settings below.
There is no secondary embedded-database switch; `XREPORT_DB_BACKEND` is the
strict selector.

SQLite ensures schema initialization at backend startup. PostgreSQL performs database and schema initialization during backend startup using `.env` connection settings.

## Interoperability

- Frontend calls backend routes through `/api`.
- Vite dev and preview proxy `/api` to `http://FASTAPI_HOST:FASTAPI_PORT`.
- The Windows launcher starts the backend, waits for `/api/health`, then starts the frontend preview and opens the configured UI URL.
- Ollama discovery uses local `/api/tags`; generation uses local `/api/chat` with image bytes encoded in the request. XREPORT never pulls an Ollama model automatically.
