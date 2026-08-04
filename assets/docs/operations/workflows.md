# Operations Workflows

Last updated: 2026-08-04

## Prepare A Dataset

1. Open the Dataset page.
2. Load or upload a dataset.
3. Run preparation and processing.
4. Confirm dataset status and metadata before training.

Expected result:

- the dataset is available in a prepared, usable state for downstream training or validation

## Train A Model

1. Open the Training page.
2. Choose dataset, checkpoint, and training parameter options.
3. Start training.
4. Monitor live progress and metrics.
5. Stop or resume when needed.

Expected result:

- checkpoints are produced and listed for later inference and validation

## Generate Reports

1. Open the Inference page and review the research-use warning.
2. Select a model and read its exact status reason, revision, adapter, loader, and declared output sections. Only `ready` models can run.
3. Add no more than the model-specific image limit and enter clinical context only when the selected contract supports it.
4. Choose a generation profile, submit the images, and poll the background job until completion or cancellation. Use Cancel generation when an active request should stop.
5. Edit the declared raw report, Findings, and/or Impression fields only; inspect returned provenance before copying or exporting. Changing model, images, or profile clears the existing draft.

Expected result:

- research-use-only draft reports are generated for qualified review; models and outputs are not clinically approved
- the four unsuitable catalogue candidates remain visible with actionable unavailable reasons
- no network fallback, gated-access flow, weight download, or manifest promotion occurs during this workflow

## Validate a cached model

Use `app/scripts/validate_inference_model.py` only with a complete exact snapshot and a public/de-identified fixture. Supply the fixture provenance, de-identification statement, and matching SHA-256 with the command; no fixture or patient data is bundled. The cache-only command runs provider and job-compatible result checks, validates declared sections and exact raw-text persistence using temporary recording, and writes metadata under `assets/QA/`. A deferred run is expected when `HF_CACHE_DIR` is unset; it does not change catalogue readiness.

## Validate Quality

1. Start dataset validation or checkpoint evaluation from validation flows.
2. Wait for completion through the polling workflow.
3. Review quality metrics and generated validation artifacts.

Expected result:

- quality indicators are available for model comparison and release decisions
