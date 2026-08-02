# Runtime Configuration

Last updated: 2026-08-02

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
- `HF_LOCAL_ONLY` (must remain `true` for embedded Hugging Face generation)
- `HF_CACHE_DIR` (existing Hugging Face cache root)
- `INFERENCE_DEVICE` (for example, `auto` or `cuda`)
- `INFERENCE_MAX_LOADED_MODELS` (must remain `1`; the embedded provider keeps one model resident)
- `INFERENCE_MODEL_TIMEOUT` (generation/model-operation timeout in seconds)

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
- Hugging Face discovery and generation resolve only the per-model cached commit declared in `settings/inference_models.json`. Network fallback is disabled, and remote code is allowed only for individually approved manifest entries. `HF_CACHE_DIR` is intentionally unset by default; this produces configured but unavailable entries without downloading anything.
- The inference catalog combines all five configured Hugging Face entries with discovered XREPORT checkpoints. It reports disabled, incompatible, gated, missing-cache, unvalidated-cache, and ready states with reasons. Only catalog entries with `ready` status can be submitted for generation.
- A ready Hugging Face entry requires an exact-revision real-inference receipt under `assets/QA/inference_validation/`; a manifest status flag or file-presence check cannot promote it. Use `app/scripts/validate_inference_model.py` with `KERAS_BACKEND=torch` when a complete snapshot and appropriate fixture already exist.
