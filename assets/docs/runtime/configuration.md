# Runtime Configuration

Last updated: 2026-10-26

## Shared Configuration Sources

- Deployment and infrastructure configuration: `settings/.env`
- Database connection configuration: `<resource root>/database-config.json`
- Runtime application settings: the singleton `application_settings` database
  row, managed through the Settings API
- Static inference catalogue: typed definitions in
  `app/server/configurations/inference_models.py`
- Tracked first-run environment template: `settings/.env.example`

The environment file contains process-level deployment and runtime policy, but
no database settings. The dedicated database configuration file contains only
the selected backend and its connection parameters; PostgreSQL passwords are
protected with Windows DPAPI. The active database stores only user-editable
application values managed through the Settings API. The reviewed model
catalogue is immutable typed application code; it is not loaded from a runtime
file. Missing or invalid database configuration opens the restricted Database
recovery UI; on first run a default SQLite configuration is created
automatically.

## Packaged desktop configuration

Packaged mode receives an internal runtime contract from Rust:
`XREPORT_RUNTIME_ROOT`, `XREPORT_DATA_ROOT`, `XREPORT_RELEASE_VERSION`,
`XREPORT_RUNTIME_VARIANT`, `XREPORT_CLIENT_DIST_DIR`, and the ephemeral
`XREPORT_DESKTOP_TOKEN`. These values are absent from `.env.example` and are
not emitted into logs. `XREPORT_RESOURCES_DIR` is ignored in packaged mode; use Settings → Storage instead.

Mutable deployment configuration is `%LOCALAPPDATA%\XREPORT\data\.env`;
database connection configuration is
`%LOCALAPPDATA%\XREPORT\data\database-config.json`. With the default SQLite
selection, application settings and other database records are stored in
`%LOCALAPPDATA%\XREPORT\data\database.db`. The immutable catalogue remains in
the extracted runtime as typed Python code. Database, checkpoints, model
downloads, tokenizers, templates, caches, and logs are all below the data root.

## Key Environment Variables

- `FASTAPI_HOST`
- `FASTAPI_PORT`
- `UI_HOST`
- `UI_PORT`
- `UI_API_BASE_URL`
- `CORS_ALLOWED_ORIGINS`
- `CORS_ALLOW_CREDENTIALS`
- `CORS_ALLOW_METHODS`
- `CORS_ALLOW_HEADERS`
- `INFERENCE_HF_LOCAL_ONLY`
- `INFERENCE_DEVICE`
- `MPLBACKEND`
- `KERAS_BACKEND`
- `HF_TOKEN` (optional; required for gated Hugging Face models such as MedGemma)
- `XREPORT_RESOURCES_DIR` (optional resource-root override; defaults to
  `data`)

On Windows normal Launch, `FASTAPI_HOST` and `FASTAPI_PORT` determine the
browser origin because FastAPI serves both the SPA and `/api`. A wildcard bind
opens through loopback. `UI_HOST`, `UI_PORT`, and `UI_API_BASE_URL` configure
Angular development and the Tauri built-bundle development server. Manual
backend development selects reload behavior in the Uvicorn command itself;
there is no `.env` reload setting.
`XREPORT_SERVE_CLIENT` and `XREPORT_LAUNCH_TOKEN` are temporary launcher-owned
process variables and are not configured in `.env`.

Database selection and connection values are edited from Settings → Database.
The supported backends are SQLite and PostgreSQL using psycopg 3. The database
file supports schema validation and atomic replacement. Stray
`EMBEDDED_DATABASE`/`DATABASE_*` lines in `.env` are ignored, are never loaded
into the process environment, and `DATABASE_URL` is unsupported; use the
Database section or the startup recovery screen to configure a backend.

The application-settings record contains the following owned values:

- `global.seed`
- `jobs.polling_interval`
- `inference.model_timeout`
- `inference.model_idle_timeout`
- `theme_preference` (the global theme selector)

`INFERENCE_HF_LOCAL_ONLY` and `INFERENCE_DEVICE` are process-level inference
policies loaded from `.env`. They are not part of the UI settings bundle or the
application-settings database row. `inference.max_loaded_models` was unused
and is no longer part of the runtime settings model.

Completed local Hugging Face inference records the requested device policy and
the effective runtime topology and dtype in provenance. For `auto`, the
Transformers/Accelerate placement is observed after loading; an available CUDA
device is not treated as used unless the loaded model is actually placed on a
CUDA device. This metadata is diagnostic evidence and does not imply clinical
quality validation.

The supported runtime editing workflow is the Settings page in the Angular
application. It uses `GET /api/settings`, partial `PATCH /api/settings`, and
`POST /api/settings/reset`. The backend validates the complete typed settings
model and commits successful updates transactionally. There is no manual JSON
editing or JSON fallback path.

Every valid non-database setting change is persisted automatically. Boolean
controls save immediately; numeric inputs debounce while the user is typing
and flush on blur, category change, or page exit. Settings writes are
serialized in the client to preserve their order across rapid edits and the
theme selector. A failed write reverts the attempted value when no newer edit
exists and displays an inline error. Reset uses the backend defaults and is
also persisted immediately.

Only these values are user-editable through Settings:

- `global.seed` (`0` through `4,294,967,295`)
- `jobs.polling_interval` (`0.25` through `60` seconds)
- `inference.model_timeout` (at least `1` second)
- `inference.model_idle_timeout` (`0` or more seconds; default `300`; `0` keeps the model loaded; XREPORT Inference only in the UI)
- `theme_preference` (`light`, `dark`, or `system`)

Changing a seed, polling interval, or inference timeout applies to newly started
work. Running jobs and generations keep the value captured at their start.
The idle timeout is read by the resident-model watcher on every check (about
every 5 seconds), so a change applies to the currently loaded model without a
restart. It never interrupts a running generation.
`INFERENCE_DEVICE`, `INFERENCE_HF_LOCAL_ONLY`, and the static model catalogue
remain runtime policy and are not exposed by the Settings API. A prior browser
theme preference is imported into the active database once; later theme
changes are stored in that database and switching databases loads its theme
choice.

`UI_API_BASE_URL` should remain `/api` for the proxied local flow. The Windows PowerShell launcher always opens a dedicated terminal for backend logs. Source mode accepts `XREPORT_RESOURCES_DIR` as an absolute path or a path relative to the repository root. Environment initialization preserves an explicit process-level value over a value in `settings/.env`, so spawned job workers inherit the same resource root resolved by the parent. When neither is set, the storage root comes from `settings/storage-location.json` (source) or `%LOCALAPPDATA%\XREPORT\data\storage-location.json` (packaged), written atomically by Settings → Storage; precedence is process environment, then `settings/.env`, then that pointer file, then the default. Packaged mode ignores that source-relative override: immutable files stay in the verified extracted runtime, while the SQLite database and all mutable state are under `%LOCALAPPDATA%\\XREPORT\\data`.

## Database switching

The Database settings section reports the active backend, tests candidate
connections, validates the schema, and applies a switch without restarting.
The Apply action stays disabled until the current draft passes its connection
test; editing the draft clears that result. The switch endpoint validates the
candidate again before activation and restores the previous connection and
configuration if activation fails. Switching pauses new API requests, drains
in-flight requests, and is blocked while a background job is running. The
selected database is initialized to the current Alembic head before activation.
The application then reloads settings from that database. Databases remain
independent; The applications do not copy records or settings between them.

If a database configuration is missing, corrupt, or unavailable at startup, a
restricted Database settings screen lets the user repair or select a backend
before normal routes are enabled. Restart the app after changing process-level
launch configuration such as `XREPORT_RESOURCES_DIR`, host, port, CORS, or
inference policy. Ordinary database switches do not require a restart. A saved
PostgreSQL password can be opened only by the Windows account that protected
it; re-enter it after moving the data directory to another account or machine.

## Interoperability

- Frontend calls backend routes through `/api`.
- Angular `start` and `dev` proxy `/api` to `http://FASTAPI_HOST:FASTAPI_PORT` using `src/proxy.conf.cjs`.
- CORS is disabled when `CORS_ALLOWED_ORIGINS` is empty. When configured, comma-separated origins, methods, headers, and credential behavior are read from `.env` at process startup; wildcard origins cannot be combined with credentials.
- Normal Windows Launch serves `dist/client-angular/browser` from FastAPI,
  applies Angular route fallback, excludes `/api` from that fallback, waits for
  `/api/health`, and opens the configured FastAPI URL. It rebuilds only when
  the deterministic frontend build-state manifest is missing or stale.
- The `preview` script serves the same built bundle with a Node static server
  and `/api` reverse proxy for local bundle validation and Tauri development.
  It is not the production Windows web server.
- Source mode derives the runtime root from the repository layout and uses the
  explicit `XREPORT_RESOURCES_DIR` override when provided. Packaged mode uses
  the runtime/data roots supplied by the Tauri shell; it does not infer them
  from the current working directory or executable location.
- The application owns all transient model/tool caches under the canonical
  `<runtime root>/runtimes/cache` path in source mode or
  `<data root>/runtimes/cache` in packaged mode. `HF_HOME`, `HF_HUB_CACHE`,
  `HF_MODULES_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`, `KERAS_HOME`, and
  `MPLCONFIGDIR` are set to subdirectories below that root at startup; hostile
  global cache values, including deprecated `TRANSFORMERS_CACHE`, are cleared.
- The external catalogue contains the reviewed SHA-pinned public report-generation models currently declared by application policy (three chest-X-ray specialists and the broader gated MedGemma option at this revision). Catalogue cardinality is not a product invariant. Their first Download or Generate action stages the pinned revision into `<resource root>/models/huggingface/staging`; only a verified snapshot that produces a non-empty report is promoted to `installed`.
- Installed metadata is stored in `<resource root>/models/huggingface/metadata`. Restarted processes load the verified local snapshot with `local_files_only=true` and do not consult unrelated global caches. Check for updates, repair, reinstall, and download-update are explicit user actions.

## Application profile variables

- `XREPORT_APP`: `xreport` (default) or `studio`. Set by the launcher; selects routers, the data
  root (`<XREPORT_RESOURCES_DIR>/studio` for Studio), and the Angular output used for source hosting.
  Packaged builds reject anything but `xreport`.
- `STUDIO_FASTAPI_PORT`: source-mode Studio backend port (default 5013, `settings/.env`).

## Storage Root

- One storage root owns `models`, `checkpoints`, `templates`, `exports`, `validation_receipts`, `logs`, `database.db`, `database-config.json` and (source mode) `studio`. Runtime extraction, shell state, `.env`, NLTK data and disposable caches stay in the fixed bootstrap location; the Tauri shell's own log files also stay there.
- Settings → Storage validates a candidate (absolute, writable, not inside the application files or the current root, enough free space, empty or already an XREPORT root, network/removable drive warning), then copies the owned entries into a staging folder, verifies sizes, rewrites the copied checkpoint registry and `database-config.json` paths, moves the staged entries into place and writes the pointer last. The previous root is never modified or deleted. A target that already holds XREPORT data is used as-is and nothing is copied. Moving is refused while jobs run, when the root is set by `XREPORT_RESOURCES_DIR`, and when the application database is not the default SQLite file in the root.
- The new root is used after the backend restarts.
