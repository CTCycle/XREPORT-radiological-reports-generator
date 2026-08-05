# Runtime Configuration

Last updated: 2026-08-05

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
- `INFERENCE_DEVICE` (for example, `auto` or `cuda`)
- `INFERENCE_MAX_LOADED_MODELS` (must remain `1`; the embedded provider keeps one model resident)
- `INFERENCE_MODEL_TIMEOUT` (generation/model-operation timeout in seconds)

`VITE_API_BASE_URL` should remain `/api` for the proxied local flow. Set `BACKEND_VISIBLE=true` to open backend logs in a dedicated terminal; the default keeps the backend window hidden.

## Database Mode Switch

- `EMBEDDED_DATABASE=true` selects SQLite.
- `EMBEDDED_DATABASE=false` selects PostgreSQL and requires the external
  connection settings below.
`EMBEDDED_DATABASE` is the strict database-mode selector.

SQLite checks the database file at backend startup and initializes only a
missing file. Existing SQLite files are not recreated, reseeded, or
cross-validated. PostgreSQL startup only verifies a connection to the
configured database; use option `3` in `start_on_windows.ps1` for explicit
database and schema initialization.

## Interoperability

- Frontend calls backend routes through `/api`.
- Vite dev and preview proxy `/api` to `http://FASTAPI_HOST:FASTAPI_PORT`.
- The Windows launcher starts the backend, waits for `/api/health`, then starts the frontend preview and opens the configured UI URL.
- The application derives the portable root from the deployed folder structure and owns all model caches under `app/resources/models`. `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, and `KERAS_HOME` are set by the backend at startup; hostile or stale user-level cache variables, including deprecated `TRANSFORMERS_CACHE`, are cleared.
- The external catalogue contains only `nathansutton/generate-cxr`, a public image-to-complete-report model. Its first Generate action performs a cloud assessment, then downloads the pinned revision into `app/resources/models/huggingface/staging`; only a verified snapshot that produces a non-empty report is promoted to `installed`.
- Installed metadata is stored in `app/resources/models/huggingface/metadata`. Restarted processes load the verified local snapshot with `local_files_only=true` and do not consult unrelated global caches. Check for updates, repair, reinstall, and download-update are explicit user actions.
