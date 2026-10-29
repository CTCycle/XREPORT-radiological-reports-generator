# Operations Workflows

Last updated: 2026-10-29

## Prepare A Dataset

1. Open the Dataset page.
2. Select the image folder and upload a table containing `text` plus a
   recognized image column.
3. Load the dataset. If unmatched rows are reported, review the matched and
   unmatched counts; choose **Import matched rows** only when a labeled
   partial import is acceptable.
4. Run preparation and processing.
5. Confirm dataset status and metadata before training.

For a non-empty dataset, fractional sampling retains at least one record so a
small dataset cannot be reduced to an empty validation or processing job.

Expected result:

- missing required columns produce a structured error without discarding the
  upload
- image identifiers are matched by filename stem without regard to letter case
- unmatched rows are never silently imported; confirmed partial imports are
  visibly labeled with their counts
- the dataset is available in a prepared, usable state for downstream training or validation
- deleting a source dataset with processed dependents returns a conflict that
  names those dependents; delete the processed datasets first

## Configure A Model Architecture (Studio)

1. Open the Model page. Presets are untrained architecture definitions;
   built-in presets are read-only and the XREPORT default (which can never be
   removed) reproduces the original architecture. Other built-in presets can be
   removed from the list and restored later; custom presets can be deleted.
2. Select a component in the diagram and adjust its parameters. Encoder and
   decoder are configured independently. Compare the draft with any preset
   (architecture only) or checkpoint (architecture and trained-weight shapes).
3. Save as a custom preset, or export/import it as JSON
   (`xreport-model-architecture`, version 1; files with the former shared
   heads/feed-forward/dropout keys still import).

Settings such as dropout, activations, layer-norm epsilon, temperature and
encoder freezing keep every tensor shape, so a checkpoint trained with one
value loads with another. Layer counts, width, heads, head size,
feed-forward and classifier units change tensor shapes: weights of a
checkpoint with different shapes cannot be loaded, and the comparison lists
the affected tensors. The pinned BEiT
encoder weights are always reused. Other image encoders were evaluated but are
not offered: XREPORT Inference only imports checkpoints built on the pinned
BEiT revision, and Studio has no offline installer for other encoders.

## Train A Model

1. Open the Training page.
2. Choose the dataset, an architecture preset, and training options.
3. Start training.
4. Monitor the independent worker phase and numeric progress. Before the first
   batch, the phase identifies preparation such as dataset, device, loader, or
   model initialization and shows elapsed time in the current phase.
5. Stop or resume when needed.

Training runs Keras 3 on the PyTorch backend in a spawned worker process, so
the model, optimizer and VRAM are released when each run ends. Options that
need care:

- **Enable GPU** off trains on the CPU even when CUDA is present (the worker
  builds and fits inside `keras.device`). A missing GPU falls back to the CPU;
  a GPU index that does not exist fails the run.
- **Mixed precision** applies on CUDA only: `mixed_bfloat16` where the GPU
  supports bfloat16, otherwise `mixed_float16` with Keras loss scaling. The
  frozen BEiT encoder runs under torch autocast.
- **Gradient accumulation steps** applies one optimizer step every N batches
  (Keras `gradient_accumulation_steps`), for an effective batch of
  batch size × N. Scheduler warm-up steps count optimizer steps.
- **Dataloader workers** decode and resize images in parallel processes; on
  Windows each worker re-imports PyTorch, so the first batch of each loader
  starts several seconds later. Use persistent workers to pay that once.
  Pinned memory is applied only for CUDA runs.
- **Save checkpoints** rewrites the run's checkpoint atomically after every
  completed epoch (`saved_model.keras` plus the configuration, metadata and
  history). A stopped or failed run then keeps its last completed epoch as a
  registered, resumable checkpoint; without a completed epoch its folder is
  removed. A custom checkpoint name that already exists is rejected (409).
  The archive is written from an on-disk staging folder rather than Keras' in-memory
  writer, which fails above 2 GB (the default preset with optimizer state is
  about 5.3 GB).

Expected result:

- checkpoints are produced and listed for later inference and validation
- a worker that exits without a success payload or stops advancing beyond its
  calibrated watchdog deadline becomes a typed recoverable failure; it does
  not remain indefinitely in `running`
- user-initiated Stop Training remains `CANCELLED`, including cancellation
  during worker initialization or before the first batch

## Generate Reports

1. Open the Inference page and review the research-use warning.
2. Select one of the current Public Models or a locally trained Custom XReport Model. Read the anatomy scope, demand, approximate size, licence/access badge, exact revision, adapter, loader, and declared output sections. A public model can be Downloaded explicitly or submitted while `not_downloaded`; Generate will prepare it in the background.
3. Add no more than the model-specific image limit and enter clinical context only when the selected contract supports it.
4. Choose a generation profile (changeable at any time; regenerating after a change creates a new saved run, and XReport checkpoints decode Deterministic and Concise identically) and, with more than one image, **Run on**: the selected image only (default, one report), each image separately (one independent report and saved row per image), or (multi-view models; disabled with a reason otherwise) all images as one study (one consolidated report). Per-image results are browsed with the report selector in step 3; each report keeps its source image, edits, save and export independently. Submit and poll the background job until completion or cancellation. Use Cancel generation when an active request should stop. Navigating away does not stop the job: returning to the page resumes progress tracking or restores the finished, saved report. In per-image runs a failed image is reported while the others still complete.
5. After completion, wait for the saved-report status, then edit the declared raw report, Findings, and/or Impression fields. Select **Save edits** to persist changes, or **View in Reports** to review the saved session. Copy or export remains available for qualified review. Changing the model or study, starting another generation, or leaving the route asks before discarding unsaved edits; browser reload/close prompts where supported.

Expected result:

- research-use-only draft reports are generated for qualified review; models and outputs are not clinically approved
- completed generated reports are already available in Reports; inference-page edits remain unsaved until **Save edits** succeeds, and the original model output remains available in report history
- failed edit saves retain the editable draft and show an error so the user can retry; study-level reports displayed across multiple images update their single persisted history row
- only currently declared pinned public report-generating models are visible in their own section; degraded public entries cannot generate, and catalogue cardinality is not a product invariant. Complete custom XREPORT checkpoints are shown separately and are never public downloads
- first-use installation reports cloud assessment, download, verification, loading, generation, and activation without blocking the UI

## Validate a local model

Use the real Generate flow with an authorized, de-identified image. The first
run installs the exact pinned revision into `<resource root>/models/huggingface`
and produces a report. When capturing validation evidence, save the scoped
metadata and provenance under `runtimes/cache/`; record the outcome and limitations
in the existing campaign or status ledger instead of creating a separate
per-slice report. A subsequent run should reuse the active snapshot without
another download. This technical check does not establish model quality.

## Validate Quality

1. Start dataset validation or checkpoint evaluation from validation flows. Text statistics can
   also count tokens per report: choose a compatible tokenizer in the wizard or the validation
   route; counts include special tokens, are stored per record and aggregated, and the tokenizer
   is recorded on the run.
2. Wait for completion through the polling workflow.
3. Review quality metrics and generated validation artifacts.

Expected result:

- quality indicators are available for model comparison and release decisions
- checkpoint evaluation uses the checkpoint-associated processed dataset and
  reports typed input/data failures instead of raw Keras exceptions
- validation, processing, training, and resume fail closed when stored image
  paths have been deleted
