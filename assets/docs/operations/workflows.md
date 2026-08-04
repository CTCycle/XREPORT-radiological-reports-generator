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
2. Select a model and read its exact installation state, revision, adapter, loader, and declared output sections. `generate-cxr` can be submitted while `not_installed`; Generate will prepare it in the background.
3. Add no more than the model-specific image limit and enter clinical context only when the selected contract supports it.
4. Choose a generation profile, submit the images, and poll the background job until completion or cancellation. Use Cancel generation when an active request should stop.
5. Edit the declared raw report, Findings, and/or Impression fields only; inspect returned provenance before copying or exporting. Changing model, images, or profile clears the existing draft.

Expected result:

- research-use-only draft reports are generated for qualified review; models and outputs are not clinically approved
- only supported report-generating models are visible; the unavailable custom XREPORT checkpoint is shown through the provider status
- first-use installation reports cloud assessment, download, verification, loading, generation, and activation without blocking the UI

## Validate a local model

Use the real Generate flow with a public/de-identified chest X-ray. The first run installs the exact pinned revision into `app/resources/models/huggingface`, records metadata and provenance under `assets/QA/` when capturing validation evidence, and generates a real report. A subsequent run must reuse the active snapshot without another download.

## Validate Quality

1. Start dataset validation or checkpoint evaluation from validation flows.
2. Wait for completion through the polling workflow.
3. Review quality metrics and generated validation artifacts.

Expected result:

- quality indicators are available for model comparison and release decisions
