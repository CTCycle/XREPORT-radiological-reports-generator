# Local Inference Models

Last updated: 2026-08-05

## Safety scope

All catalogue models and generated reports are for research use only. They are not clinically approved and require independent review.

## Supported catalogue

`GET /api/inference/models` lists the public `huggingface:nathansutton/generate-cxr` model plus any complete custom `xreport:` checkpoints discovered under `app/resources/checkpoints`. The external model accepts one chest X-ray and an optional indication and returns one complete raw report. It is the only external option advertised as runnable.

MedGemma was removed because it is gated and cannot be installed anonymously. MAIRA-2 was removed because it is gated, outside the supported Transformers range, and does not provide this application's complete report contract. CheXagent Impression was removed because it produces impression-only output and requires an unsupported custom stack. CXRMate-2 was removed because its large custom-code runtime is not implemented or approved here.

## Project-local lifecycle

The backend derives the portable application root and creates this structure at startup:

```text
app/resources/
├── checkpoints/
├── models/
│   ├── huggingface/
│   │   ├── installed/<model>/<revision>/
│   │   ├── staging/<operation>/<model>/<revision>/
│   │   ├── rollback/<model>/<revision-or-operation>/
│   │   ├── metadata/<model>.json
│   │   └── hub-cache/
│   ├── tokenizers/
│   ├── XRAYEncoder/
│   ├── torch/
│   └── keras/
├── templates/
├── logs/
└── database.db
```

`HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, and `KERAS_HOME` are application-owned cache settings. Deprecated or user-level cache variables are cleared, and local model loading always uses the verified snapshot path with `local_files_only=true`.

On first Generate, the service validates the uploaded image, records the cloud assessment, and starts a cancellable background installation. The exact manifest revision and approved files are downloaded to staging. File sizes, SHA-256 values, configuration, processor initialization, and model initialization are checked before a candidate can be used. A candidate is promoted to `installed` only after it produces a non-empty report for the current study. Cancellation and network errors during first-use installation leave resumable staging files and metadata but never expose them as active. If a working active revision already exists, a canceled maintenance or reinstall operation discards its partial staging immediately; retries start clean while the active revision remains available.

The active revision is reused on later Generate calls and after restart without a network lookup or download. Existing working revisions are never replaced automatically. The model details panel exposes Check for updates, Repair installation, Reinstall model, and Download update. Updates are staged beside the active revision and the old revision is retained under `rollback` until a successful real inference activates the candidate.

## Maintenance API

- `POST /api/inference/models/check-update` with `{ "model_ref": "huggingface:nathansutton/generate-cxr" }` checks the current upstream commit on demand.
- `POST /api/inference/models/maintenance` accepts `repair`, `reinstall`, or `download_update` and returns a normal background job contract.
- Job status results include a typed `lifecycle` object with phase, status text, current file, byte/file progress, revision, and project-relative paths.

## Custom XREPORT

XREPORT discovery remains separate and only complete trained checkpoints are selectable. The repository currently has no trained custom XReport checkpoint; incomplete training fixtures are not advertised as usable and do not block external model validation.
