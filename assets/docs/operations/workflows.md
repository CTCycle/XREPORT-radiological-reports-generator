# Operations Workflows

Last updated: 2026-07-15

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

1. Open the Inference page.
2. Select a ready model from the local model catalog and choose a generation profile.
3. Submit image inputs and supported clinical context for inference.
4. Poll job status until completion.
5. Review generated text outputs.

Expected result:

- research-use-only draft reports are generated for qualified review; models and outputs are not clinically approved

## Validate Quality

1. Start dataset validation or checkpoint evaluation from validation flows.
2. Wait for completion through the polling workflow.
3. Review quality metrics and generated validation artifacts.

Expected result:

- quality indicators are available for model comparison and release decisions
