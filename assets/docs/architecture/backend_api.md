# XREPORT Backend API

Last updated: 2026-06-03

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

- `GET /api/inference/checkpoints`
- `POST /api/inference/generate`
- `GET /api/inference/jobs/{job_id}`
- `DELETE /api/inference/jobs/{job_id}`

## Root Behavior

- When `app/client/dist` is available, the backend serves SPA files from the built frontend bundle.
- Otherwise, `GET /` redirects to `/docs`.
