# Runtime Configuration

Last updated: 2026-09-17

## Shared Configuration Sources

- Deployment and infrastructure configuration: `settings/.env`
- Runtime application settings: the singleton `application_settings` database
  row, managed through the Settings API
- Static inference catalogue: typed definitions in
  `app/server/configurations/inference_models.py`
- Tracked first-run environment template: `settings/.env.example`

The environment file is not a second application-settings store. Deployment,
database connection, and secret values continue to come from `settings/.env`.
The four user-editable application values and the two hidden runtime policy
values are read from the database. The reviewed model catalogue is immutable
typed application code; it is not loaded from a runtime file.

The Alembic upgrade that introduced `application_settings` is the only legacy
reader for the former JSON application configuration. It imports the existing
values once, then the obsolete file is removed. Normal startup, the Settings
API, desktop packaging, and runtime services do not read or recreate it.

## Packaged desktop configuration

Packaged mode receives an internal runtime contract from Rust:
`XREPORT_RUNTIME_ROOT`, `XREPORT_DATA_ROOT`, `XREPORT_RELEASE_VERSION`,
`XREPORT_RUNTIME_VARIANT`, `XREPORT_CLIENT_DIST_DIR`, and the ephemeral
`XREPORT_DESKTOP_TOKEN`. These values are absent from `.env.example` and are
not emitted into logs. `XREPORT_RESOURCES_DIR` is ignored in packaged mode.

Mutable deployment configuration is `%LOCALAPPDATA%\XREPORT\data\.env` and
mutable application settings are stored in the database at
`%LOCALAPPDATA%\XREPORT\data\database.db` when SQLite is selected. The
immutable catalogue remains in the extracted runtime as typed Python code.
Database, checkpoints, model downloads, tokenizers, templates, caches, and
logs are all below the data root.

## Key Environment Variables

- `FASTAPI_HOST`
- `FASTAPI_PORT`
- `UI_HOST`
- `UI_PORT`
- `UI_API_BASE_URL`
- `RELOAD`
- `BACKEND_VISIBLE`
- `ALWAYS_REBUILD` (set to `true` to rebuild the frontend whenever the Windows
  launcher starts the application; defaults to `false`)
- `MPLBACKEND`
- `KERAS_BACKEND`
- `EMBEDDED_DATABASE` (`true` for SQLite or `false` for PostgreSQL)
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
- `HF_TOKEN` (optional; required for gated Hugging Face models such as MedGemma)
- `XREPORT_RESOURCES_DIR` (optional resource-root override; defaults to
  `app/resources`)

`DATABASE_URL` is intentionally unsupported. External database mode requires
the decomposed `DATABASE_ENGINE`, `DATABASE_HOST`, `DATABASE_PORT`,
`DATABASE_NAME`, and `DATABASE_USERNAME` values.

The application-settings record contains the following owned values:

- `global.seed`
- `features.allow_local_filesystem_access`
- `jobs.polling_interval`
- `inference.model_timeout`

The same database row also stores the hidden process policy values
`inference.hf_local_only` and `inference.device`. They are loaded at startup,
are not returned by the public Settings API, and are not user-editable.
`inference.max_loaded_models` was unused and is no longer part of the runtime
settings model.

The supported runtime editing workflow is the Settings page in the Angular
application. It uses `GET /api/settings`, partial `PATCH /api/settings`, and
`POST /api/settings/reset`. The backend validates the complete typed settings
model and commits successful updates transactionally. There is no manual JSON
editing or JSON fallback path.

Only these values are user-editable through Settings:

- `global.seed` (`0` through `4,294,967,295`)
- `features.allow_local_filesystem_access`
- `jobs.polling_interval` (`0.25` through `60` seconds)
- `inference.model_timeout` (at least `1` second)

Changing a seed, polling interval, or inference timeout applies to newly started
work. Running jobs and generations keep the value captured at their start.
`inference.device`, `inference.hf_local_only`, and
the static model catalogue remain runtime policy and are not exposed by the
Settings API. Theme selection remains a frontend-local preference.

`UI_API_BASE_URL` should remain `/api` for the proxied local flow. Set `BACKEND_VISIBLE=true` to open backend logs in a dedicated terminal; the default keeps the backend window hidden. Source mode accepts `XREPORT_RESOURCES_DIR` as an absolute path or a path relative to the repository root. Packaged mode ignores that source-relative override: immutable files stay in the verified extracted runtime, while the SQLite database and all mutable state are under `%LOCALAPPDATA%\\XREPORT\\data`.

## Database Mode Switch

- `EMBEDDED_DATABASE=true` selects SQLite.
- `EMBEDDED_DATABASE=false` selects PostgreSQL and requires the external
  connection settings below.
`EMBEDDED_DATABASE` is the strict database-mode selector.

SQLite checks the database file at backend startup and initializes only a
missing file. Existing SQLite files are not recreated, reseeded, or silently
adopted when their migration state is absent or incompatible. PostgreSQL
startup only verifies a connection to the configured database; use option `4`
in `start_on_windows.ps1` for explicit database and schema initialization.

## Interoperability

- Frontend calls backend routes through `/api`.
- Angular dev and preview proxy `/api` to `http://FASTAPI_HOST:FASTAPI_PORT` using `src/proxy.conf.cjs`.
- The Windows launcher starts the backend, waits for `/api/health`, then starts the frontend preview and opens the configured UI URL.
- Source mode derives the runtime root from the repository layout and uses the
  explicit `XREPORT_RESOURCES_DIR` override when provided. Packaged mode uses
  the runtime/data roots supplied by the Tauri shell; it does not infer them
  from the current working directory or executable location.
- The application owns all model caches under `<resource root>/models`.
  `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, and `KERAS_HOME` are set by the
  backend at startup; hostile or stale user-level cache variables, including
  deprecated `TRANSFORMERS_CACHE`, are cleared.
- The external catalogue contains exactly five SHA-pinned public report-generation models (four chest-X-ray specialists and the broader gated MedGemma option). Their first Download or Generate action stages the pinned revision into `<resource root>/models/huggingface/staging`; only a verified snapshot that produces a non-empty report is promoted to `installed`.
- Installed metadata is stored in `<resource root>/models/huggingface/metadata`. Restarted processes load the verified local snapshot with `local_files_only=true` and do not consult unrelated global caches. Check for updates, repair, reinstall, and download-update are explicit user actions.
