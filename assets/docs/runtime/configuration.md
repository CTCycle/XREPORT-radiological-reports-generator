# Runtime Configuration

Last updated: 2026-07-23

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
- `ALWAYS_REBUILD` (set to `true` to rebuild the frontend whenever the Windows
  launcher starts the application; defaults to `false`)
- `MPLBACKEND`
- `KERAS_BACKEND`
- `EMBEDDED_DATABASE` (`true` for SQLite or `false` for PostgreSQL)
- `DATABASE_URL`
- `DATABASE_ENGINE` (`postgres`, `postgresql`, `postgresql+psycopg`, or
  `postgresql+psycopg2` when external mode is selected)
- `DATABASE_HOST`
- `DATABASE_PORT`
- `DATABASE_NAME`
- `DATABASE_USERNAME`
- `DATABASE_PASSWORD`
- `DATABASE_SSL`
- `DATABASE_SSL_CA`
- `DATABASE_CONNECT_TIMEOUT`
- `DATABASE_INSERT_BATCH_SIZE`
- `XREPORT_OLLAMA_BASE_URL` (loopback local Ollama endpoint)
- `XREPORT_OLLAMA_KEEP_ALIVE` (model residency passed to `/api/chat`, default `5m`)
- `XREPORT_INFERENCE_MODEL_TIMEOUT` (generation read timeout in seconds)
- `XREPORT_HF_LOCAL_ONLY` (must remain `true` for Hugging Face generation)
- `XREPORT_HF_CACHE_DIR` (existing Hugging Face cache root)
- `XREPORT_HF_MEDGEMMA_REVISION` (exact 40-character cached commit)
- `XREPORT_INFERENCE_MAX_LOADED_MODELS` (minimum 1; default comes from static inference configuration)
- `XREPORT_INFERENCE_MODEL_TIMEOUT` (generation/model-operation timeout in seconds)

`VITE_API_BASE_URL` should remain `/api` for the proxied local flow. Set `BACKEND_VISIBLE=true` to open backend logs in a dedicated terminal; the default keeps the backend window hidden.

## Database Mode Switch

- `EMBEDDED_DATABASE=true` selects SQLite.
- `EMBEDDED_DATABASE=false` selects PostgreSQL and requires the external
  connection settings below.
`EMBEDDED_DATABASE` is the strict database-mode selector.

SQLite ensures schema initialization at backend startup. PostgreSQL performs database and schema initialization during backend startup using `.env` connection settings.

## Interoperability

- Frontend calls backend routes through `/api`.
- Vite dev and preview proxy `/api` to `http://FASTAPI_HOST:FASTAPI_PORT`.
- The Windows launcher starts the backend, waits for `/api/health`, then starts the frontend preview and opens the configured UI URL.
- Ollama discovery uses local `/api/tags`; generation uses local `/api/chat` with image bytes encoded in the request. XREPORT never pulls an Ollama model automatically.
- Hugging Face discovery and generation resolve only the configured cached MedGemma commit. Network fallback and remote code are disabled.
- The inference catalog combines configured Ollama/Hugging Face entries with discovered XREPORT checkpoints. Only catalog entries with `ready` status can be submitted for generation.
