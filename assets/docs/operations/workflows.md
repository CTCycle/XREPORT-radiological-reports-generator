# Operations Workflows

Last updated: 2026-08-21

## Prepare A Dataset

1. Open the Dataset page.
2. Select the image folder and upload a table containing `text` plus a
   recognized image column.
3. Load the dataset. If unmatched rows are reported, review the matched and
   unmatched counts; choose **Import matched rows** only when a labeled
   partial import is acceptable.
4. Run preparation and processing.
5. Confirm dataset status and metadata before training.

Expected result:

- missing required columns produce a structured error without discarding the
  upload
- unmatched rows are never silently imported; confirmed partial imports are
  visibly labeled with their counts
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
2. Select one of the five Public Models or a locally trained Custom XReport Model. Read the anatomy scope, demand, approximate size, licence/access badge, exact revision, adapter, loader, and declared output sections. A public model can be Downloaded explicitly or submitted while `not_downloaded`; Generate will prepare it in the background.
3. Add no more than the model-specific image limit and enter clinical context only when the selected contract supports it.
4. Choose a generation profile, submit the images, and poll the background job until completion or cancellation. Use Cancel generation when an active request should stop.
5. Edit the declared raw report, Findings, and/or Impression fields only; inspect returned provenance before copying or exporting. Changing model, images, or profile clears the existing draft.

Expected result:

- research-use-only draft reports are generated for qualified review; models and outputs are not clinically approved
- exactly five pinned public report-generating models are visible in their own section; complete custom XREPORT checkpoints are shown separately and are never public downloads
- first-use installation reports cloud assessment, download, verification, loading, generation, and activation without blocking the UI

## Validate a local model

Use the real Generate flow with a public/de-identified chest X-ray. The first run installs the exact pinned revision into `<resource root>/models/huggingface`, records metadata and provenance under `assets/QA/` when capturing validation evidence, and generates a real report. A subsequent run must reuse the active snapshot without another download.

## Validate Quality

1. Start dataset validation or checkpoint evaluation from validation flows.
2. Wait for completion through the polling workflow.
3. Review quality metrics and generated validation artifacts.

Expected result:

- quality indicators are available for model comparison and release decisions
- checkpoint evaluation uses the checkpoint-associated processed dataset and
  reports typed input/data failures instead of raw Keras exceptions
- validation, processing, training, and resume fail closed when stored image
  paths have been deleted
