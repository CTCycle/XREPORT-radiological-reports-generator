# XREPORT Backend API

Last updated: 2026-10-27

Each application has a checked-in OpenAPI snapshot, `app/shared/openapi.xreport.json` and
`app/shared/openapi.studio.json`, generated from `create_app(profile).openapi()`. Regenerate both
from the repository root with:

```powershell
& ".\app\server\.venv\Scripts\python.exe" app/scripts/export_openapi.py
```

then run `npm run generate:api` in `app/client`. The XREPORT Inference application mounts the inference,
jobs, settings, and database-settings routers plus `POST /api/exchange/import`; XREPORT Studio mounts
upload, preparation, model, training, validation, jobs, settings, database settings, `POST /api/exchange/export`,
and `GET /api/exchange/exports/{filename}`. Endpoint groups below that belong to the other
application return 404 (see [applications and checkpoint exchange](applications_and_checkpoint_exchange.md)).

All routers are mounted under `/api`.

## Upload

- `POST /api/upload/dataset`

The upload response contains an explicit `upload_id`. The client must send that
identifier to `POST /api/preparation/dataset/load`; the backend never selects a
latest or otherwise implicit upload.

## Generic Jobs

- `GET /api/jobs?job_type={job_type}&status={status}`
- `GET /api/jobs/{job_id}`
- `DELETE /api/jobs/{job_id}`

The query parameters are optional filters. Every long-running preparation,
training, validation, evaluation, and inference start endpoint returns a
`job_id`; clients poll the generic job resource and use `DELETE` to request
cancellation. The client keeps an active job visible while a transient status
request fails; an explicit missing-job response or a terminal job status ends
polling. Cancellation remains active until the backend reports a terminal
state. There are no feature-specific job routes and no separate training-status
endpoint.

## Preparation

- `GET /api/preparation/dataset/status`
- `GET /api/preparation/dataset/names`
- `GET /api/preparation/dataset/processed/names`
- `GET /api/preparation/dataset/metadata/{dataset_name}`
- `DELETE /api/preparation/dataset/{dataset_name}`
- `POST /api/preparation/images/validate`
- `POST /api/preparation/dataset/load`
- `POST /api/preparation/dataset/process`
- `GET /api/preparation/dataset/{dataset_name}/images/count`
- `GET /api/preparation/dataset/{dataset_name}/images/{index}`
- `GET /api/preparation/dataset/{dataset_name}/images/{index}/content`
- `POST /api/preparation/folder/select`

`POST /api/preparation/dataset/load` requires the uploaded table to contain
`text` and a recognized image column (`image`, `filename`, `file`, `img`, or
`image_name`). Missing columns return a structured 400 response and the
uploaded table remains available for correction. When some rows do not match
images, the first request returns `requires_confirmation=true` with matched
and unmatched counts without writing records. A repeat request with
`confirm_unmatched=true` imports only the matched rows and returns
`partial_import=true`; an empty match is rejected.

## Model architecture (Studio)

- `GET /api/model/encoders`: registered pretrained image encoders (BEiT only).
- `GET /api/model/presets`: built-in presets (`xreport-default`, `compact`,
  `balanced`, `large`) followed by custom presets (`custom-<id>`), each with
  analytic parameter and memory statistics.
- `POST /api/model/presets`, `PUT /api/model/presets/{key}`,
  `DELETE /api/model/presets/{key}`: custom presets only. Built-in names are
  reserved and names are unique case-insensitively (409).
- `POST /api/model/preview`: validates a raw architecture and returns field
  issues instead of a 422, plus the architecture graph, statistics sized
  against an optional processed dataset, and compatibility with an optional
  `reference` (a preset key or `checkpoint:<name>`).
- `GET /api/model/checkpoints/{checkpoint}/architecture`: the architecture a
  checkpoint was trained with, reconstructed from its configuration (legacy
  flat keys are flagged with `legacy: true`).

The graph and parameter counts come from `server/models/training/architecture.py`,
the same module the training builder uses; unit tests compare the analytic
counts with built Keras layers.

## Training

- `GET /api/training/checkpoints`
- `GET /api/training/checkpoints/{checkpoint}/metadata`
- `DELETE /api/training/checkpoints/{checkpoint}`
- `POST /api/training/start`: takes `architecture_preset` (a preset key) instead
  of individual architecture fields; an unknown key returns 400.
- `POST /api/training/resume`

## Validation

- `POST /api/validation/run`
- `POST /api/validation/checkpoint`
- `GET /api/validation/checkpoint/reports/{checkpoint}`
- `GET /api/validation/reports/{dataset_name}`

Dataset validation accepts only `text_statistics`, `image_statistics`, and
`pixels_distribution`. Checkpoint evaluation accepts only `evaluation_report`
and `bleu_score`; unsupported metric names are rejected with HTTP 422 before a
job is created. When a request omits `seed`, checkpoint evaluation uses the
configured `global.seed` value.

## Inference

- `GET /api/inference/models`
- `POST /api/inference/models/check-update`
- `POST /api/inference/models/maintenance`
- `POST /api/inference/runtime/release`
- `POST /api/inference/generate`
- `GET /api/inference/history`
- `GET /api/inference/history/{request_id}`
- `PATCH /api/inference/history/{request_id}`
- `DELETE /api/inference/history/{request_id}`

`POST /api/inference/generate` is multipart and accepts only `model_ref`, `generation_profile`, `clinical_context`, and `images`. Model readiness, capabilities, and input semantics come from `GET /api/inference/models`.

`POST /api/inference/runtime/release` unloads the resident model and returns cached CUDA blocks to the driver; it answers `{released, model_ref}`. It returns 409 while an inference or model-maintenance job runs, or while a run holds the residency lock, so it can never interrupt a generation. It does not touch inference history or client-side studies.

The multipart metadata is parsed into the domain `InferenceGenerateRequest`; the uploaded files remain typed `UploadFile` inputs and the flat field names remain unchanged.

The Angular client maps these endpoint groups to `InferenceApiService`, `DatasetApiService`, `TrainingApiService`, `ValidationApiService`, and the generic `JobsApiService`. Shared request execution and error formatting live in `ApiRequestService`; feature clients preserve the existing result/error envelope.

The inference service accepts at most 16 images and a 64 MiB total image payload. It rejects models that are absent or not ready in the catalog, unsupported clinical context, unsupported providers, invalid image types, and invalid model-specific image counts.

Expected service-layer failures use typed errors and are translated centrally into the existing `{"detail": ...}` response envelope. Background job failures remain visible through the job status response, including persistence failures during inference history writes.

Inference history is backed by the persisted `InferenceRun` and
`InferenceReport` records. The list route supports `model_ref`, lifecycle
`status`, `sort`, `limit`, and `offset` filters and returns report summaries;
the detail route returns session metadata and each generated report. PATCH
updates report drafts atomically by `image_index`, preserving the generated
text alongside an optional edited value and timestamp. DELETE removes one
session and its child reports through the existing inference-run cascade.

Dataset-consuming jobs validate every stored image path before processing. A
deleted or missing path fails closed with the `dataset_integrity_failed` job
code. Checkpoint evaluation resolves validation data from the checkpoint's
recorded `dataset_name`; unavailable data reports `checkpoint_dataset_unavailable`
and incompatible model/data shapes report `checkpoint_input_mismatch`.

## Health

- `GET /api/health`

## System Status

- `GET /api/system/status` (both profiles) reports RAM use, CUDA availability and
  device name, and GPU VRAM use in GB from live `psutil`/`torch` readings. Metrics
  that cannot be read are `null`; the endpoint does not raise. The client polls it
  every 5 s while the tab is visible and treats a failed request as backend offline.

## Runtime Settings

- `GET /api/settings`
- `PATCH /api/settings`
- `POST /api/settings/reset`

The Settings API exposes only the runtime-managed application values:
`global.seed`, `jobs.polling_interval`, `inference.model_timeout`,
`inference.model_idle_timeout`, and `theme_preference`. PATCH requests are
nested partial updates and are validated by the backend before the authoritative
`application_settings` row is updated transactionally. Reset restores those
public values while preserving static inference policy, file-backed
database connection configuration, and the model catalogue.

Database credentials, `.env` values, Hugging Face tokens, runtime paths,
`inference.device`, `inference.hf_local_only`, and
the obsolete `inference.max_loaded_models` key are never returned by this API.

## Database Settings

- `GET /api/settings/database` reports the active connection, schema readiness,
  background-job state, and recovery status without exposing credentials.
- `GET /api/settings/storage` reports the active and pending storage root, its source and folder sizes; `POST /api/settings/storage/validate` checks a candidate folder; `POST /api/settings/storage/apply` starts a `storage_migration` job (copy, or `use_existing`) that takes effect after restart. XREPORT profile only.
- `POST /api/inference/generate` accepts an optional `batch_mode` (`study` default, or `per_image`, which runs each image independently, tolerates per-image failures reported in `provenance.image_errors`, and allows up to 16 images regardless of the model's study limit).
- `GET`/`PUT`/`DELETE /api/settings/huggingface` report, validate-and-save, or remove the
  Hugging Face access token used for gated models; responses expose only `configured` and
  `source`, never the token.
- `POST /api/settings/database/test` tests a candidate connection and reports
  connection and schema readiness separately.
- `POST /api/settings/database/switch` prepares the target schema, pauses and
  drains in-flight API requests, rejects active background jobs, then activates
  the target connection. It returns application settings loaded from the
  selected database. A failed activation rolls back to the previous
  connection.

Only the database settings routes remain available during startup recovery;
other `/api` requests return 503 until a valid database is activated. SQLite
and PostgreSQL (psycopg 3) are the supported backends.

The health response reports backend status, application version, and active
database mode. It can report `database_recovery` while the restricted Database
settings UI is available. The health route is excluded from OpenAPI so launcher
readiness checks do not appear as an interactive application operation.

## Root Behavior And Serving

- By default, source-mode `GET /` redirects to `/docs`; manual source workflows
  keep this behavior and run Angular CLI for the UI.
- Normal Windows Launch opts into serving the built Angular bundle from FastAPI
  with `XREPORT_SERVE_CLIENT=1`. The root serves `index.html`, existing files
  are served as static assets, and extensionless client routes fall back to the
  SPA entrypoint.
- The source-mode SPA fallback never handles `/api` or `/api/...`; unknown API
  paths remain JSON API responses. Angular uses relative `/api` calls, so the
  production browser flow is same-origin.
- Tauri release continues to load the frontend from the bundled `ui/` assets.
  Rust continues to own the sidecar lifecycle and packaged security contract.
