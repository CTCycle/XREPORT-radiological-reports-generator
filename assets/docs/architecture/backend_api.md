# XREPORT Backend API

Last updated: 2026-07-20

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
- `POST /api/inference/generate`
- `GET /api/inference/jobs/{job_id}`
- `DELETE /api/inference/jobs/{job_id}`

`POST /api/inference/generate` is multipart and accepts only `model_ref`, `generation_profile`, `clinical_context`, and `images`. The obsolete inference checkpoint endpoint and `checkpoint`/`generation_mode` request fields are removed. Model readiness, capabilities, and input semantics come from `GET /api/inference/models`.

The inference service accepts at most 16 images and a 64 MiB total image payload. It rejects models that are absent or not ready in the catalog, unsupported clinical context, unsupported providers, invalid image types, and invalid model-specific image counts.

Expected service-layer failures use typed errors and are translated centrally into the existing `{"detail": ...}` response envelope. The mapping includes 400, 403, 404, 409, 413, 500, and 501 responses.

## Root Behavior

- When `app/client/dist` is available, the backend serves SPA files from the built frontend bundle.
- Otherwise, `GET /` redirects to FastAPI's `/docs` endpoint.
