# Local Inference Models

Last updated: 2026-08-30

## Safety scope

All catalogue models and generated reports are research-use drafts. They are not
clinically approved and require qualified review and independent verification.

## Public catalogue (schema 3)

`GET /api/inference/models` always exposes exactly five public entries. Each
entry is pinned to the commit recorded in `settings/inference_models.json`; the
catalogue is not changed when local files are removed.

| Model | Positioning | Demand and storage | Anatomy / access |
| --- | --- | --- | --- |
| `aehrc/cxrmate-multi-tf` (`330721b9aa5bba201a3eb88eba4dd9a6607f3e7a`) | Lightweight multi-view chest reporter | Low; about 0.1B parameters and 451 MB selected weights | Chest radiographs; Apache-2.0; open |
| `aehrc/cxrmate-ed` (`68251c7605067ddbea330413aade032713fd2192`) | Compact context-aware chest reporter | Low; about 0.2B parameters and 793 MB selected weights | Chest radiographs plus optional indication/history; Apache-2.0; open |
| `StanfordAIMI/CheXOne` (`0c350e6852ea08f9d9baf3b7595c1a10d4849927`) | Higher-capability vision-language chest model | High; about 4B parameters and 8.15 GB selected weights | Chest radiographs; CC-BY-NC-4.0 research licence; open |
| `aehrc/cxrmate-2` (`aa8e2d16470e20671acf049687b4707c9bf2f2b5`) | Flagship specialist with structured findings/impression | Very high; about 3B parameters and 13.31 GB full-precision weights | Chest radiographs; Apache-2.0; open |
| `google/medgemma-1.5-4b-it` (`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`) | Broader medical-imaging baseline | High; about 4B parameters and 8.64 GB selected weights | Broader medical imaging, not validated for every anatomy; Health AI Developer Foundations terms; gated |

The first four entries are chest-X-ray specialists. MedGemma is intentionally
labelled as the broader option rather than as a universally validated
radiography model. The UI shows demand, approximate size, licence, anatomy
scope, and access policy before a model is selected.

The public entries use focused adapters behind one study-level provider
contract: load and validate the pinned snapshot, preprocess the complete study,
generate, normalize findings/impression or raw-report output, and return the
common `ProviderGenerationResult`. CXRMate adapters retain their published
multi-view/section-decoding contracts; CheXOne and MedGemma use the shared
chat-style vision-language path. Custom remote code is imported only from the
integrity-verified local snapshot.

### CXRMate-ED validation status

CXRMate-ED remains selectable and runnable for research use, but its catalog
`validation_status` is currently `degraded`. The real three-case canary reaches
the image tensor, clinical context, and selected generation settings, yet two
of the three supplied fixtures still produce identical report text. The UI
shows the warning before generation, and every generated provenance record
contains `validation_status=degraded`, the warning text, and
`quality_warnings=["sensitivity_canary_failed"]`. The latest canary record is
stored under `assets/QA/inference_validation_runs/`; a stale passing receipt
cannot promote this model while the manifest remains degraded.

## Project-local lifecycle

The backend owns this structure under the configured resource root (default
`app/resources`; override with `XREPORT_RESOURCES_DIR`):

```text
app/resources/
├── checkpoints/                         # custom XREPORT training outputs
├── models/huggingface/
│   ├── installed/<model>/<revision>/    # active verified snapshot
│   ├── staging/<operation>/<model>/<revision>/
│   ├── rollback/<model>/<revision>/
│   ├── metadata/<model>.json             # lifecycle and integrity metadata
│   └── hub-cache/<model-cache>/           # model-specific Hub cache
├── tokenizers/
├── XRAYEncoder/
├── torch/
└── keras/
```

The manifest policy, local installation receipt, and effective runtime
readiness are separate. The API reports the effective `status`, explicit
`installation_state` and `integrity_status`, manifest `validation_status`,
real-inference `validation_receipt_status`, and the permitted
`available_actions`. It does not duplicate those actions as independent
boolean fields.

An explicit Download, or the first Generate, stages only approved files at the
pinned revision. Downloads resume incomplete files and publish cancellable
progress. The staged snapshot is size/hash verified and recorded as
`downloaded_unvalidated`; it becomes `ready` only after a real non-empty,
well-formed image-to-report inference succeeds. Subsequent calls and restarts
reuse the verified active snapshot with `local_files_only=true`. A failed or
cancelled first use remains resumable; a failed maintenance operation never
replaces a working active revision.

Delete local files is explicit and confirmation-gated. The runtime lock refuses
deletion while inference is active, unloads an idle resident model, and removes
only that public repository's active, candidate/staging, rollback, metadata, and
model-specific Hub-cache paths. The response reports bytes reclaimed. The JSON
catalogue and all custom XREPORT checkpoints remain untouched, so the same
public card returns to `not_downloaded` and can be downloaded again.

## Gated MedGemma access

Accept the terms on the [MedGemma model page](https://huggingface.co/google/medgemma-1.5-4b-it)
and configure `HF_TOKEN` in the backend environment, or use the standard local
Hugging Face credential store. The token is resolved only by the backend for
Hub metadata/download requests; it is never entered, returned, or logged by
the browser. Missing credentials or unaccepted terms produce a structured
`access_required` failure with a safe message and the model access link.

## Maintenance and failure contracts

- `POST /api/inference/models/check-update` checks a selected public repository.
- `POST /api/inference/models/maintenance` supports `download`, `repair`,
  `reinstall`, `download_update`, and `delete_local`; every operation is a
  background job with lifecycle and byte/file progress.
- Poll maintenance jobs through `GET /api/jobs/{job_id}` and request
  cancellation through `DELETE /api/jobs/{job_id}`.
- Job failures contain a stable code, safe message, phase, and recoverability
  for access denial, download/integrity errors, incomplete snapshots, missing
  dependencies, unsupported hardware, model-load failures, cancellation, and
  inference failures. Secrets and local credential paths are redacted.

## Custom XREPORT

Complete checkpoint artifacts are registered in the database and listed in the
persistent **Custom XReport Models** section. The filesystem stores the
registered Keras/BEiT artifacts; it is not scanned to invent checkpoint
identity. Registered artifacts that later become incomplete are surfaced as
`runtime_unavailable`, while unregistered directories are not listed. Custom
checkpoints are locally trained and never presented as public downloads. The
public model catalogue and its deletion lifecycle cannot remove or alter these
checkpoints.
