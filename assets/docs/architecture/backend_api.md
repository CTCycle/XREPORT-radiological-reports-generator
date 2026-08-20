# XREPORT Backend API

Last updated: 2026-08-20

The checked-in shared OpenAPI schema is `app/shared/openapi.json`. It mirrors the runtime FastAPI schema and is the contract snapshot available to frontend and tooling consumers. Regenerate it from the repository root with:

```powershell
$env:PYTHONPATH = "app"
& ".\\app\\server\\.venv\\Scripts\\python.exe" -c "import json; from pathlib import Path; from server.app import app; Path('app/shared/openapi.json').write_text(json.dumps(app.openapi(), indent=2) + '\\n', encoding='utf-8')"
```

All routers are mounted under `/api`.

## Upload

- `POST /api/upload/dataset`

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
- `GET /api/preparation/jobs/{job_id}`
- `DELETE /api/preparation/jobs/{job_id}`
- `GET /api/preparation/browse`

## Training

- `GET /api/training/checkpoints`
- `GET /api/training/checkpoints/{checkpoint}/metadata`
- `DELETE /api/training/checkpoints/{checkpoint}`
- `GET /api/training/status`
- `POST /api/training/start`
- `POST /api/training/resume`
- `GET /api/training/jobs/{job_id}`
- `DELETE /api/training/jobs/{job_id}`

## Validation

- `POST /api/validation/run`
- `POST /api/validation/checkpoint`
- `GET /api/validation/checkpoint/reports/{checkpoint}`
- `GET /api/validation/reports/{dataset_name}`
- `GET /api/validation/jobs/{job_id}`
- `DELETE /api/validation/jobs/{job_id}`

## Inference

- `GET /api/inference/models`
- `POST /api/inference/models/check-update`
- `POST /api/inference/models/maintenance`
- `POST /api/inference/generate`
- `GET /api/inference/jobs/{job_id}`
- `DELETE /api/inference/jobs/{job_id}`

`POST /api/inference/generate` is multipart and accepts only `model_ref`, `generation_profile`, `clinical_context`, and `images`. Model readiness, capabilities, and input semantics come from `GET /api/inference/models`.

The Angular client maps these endpoint groups to `InferenceApiService`, `DatasetApiService`, `TrainingApiService`, and `ValidationApiService`. Shared request execution and error formatting live in `ApiRequestService`; feature clients preserve the existing result/error envelope.

The inference service accepts at most 16 images and a 64 MiB total image payload. It rejects models that are absent or not ready in the catalog, unsupported clinical context, unsupported providers, invalid image types, and invalid model-specific image counts.

Expected service-layer failures use typed errors and are translated centrally into the existing `{"detail": ...}` response envelope. Background job failures remain visible through the job status response, including persistence failures during inference history writes.

## Health

- `GET /api/health`

The health response reports backend status, application version, and active database mode. It is excluded from the OpenAPI schema so launcher readiness checks do not appear as an interactive application operation.

## Root Behavior And Serving

- `GET /` on the FastAPI process redirects to `/docs`.
- The FastAPI process does not own the Angular document root in the current runtime topology.
- Angular is served by its own development or production frontend runtime, coordinated with FastAPI by `start_on_windows.ps1`.
