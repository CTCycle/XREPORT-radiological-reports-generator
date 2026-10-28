# XREPORT Project Status Ledger

Last updated: 2026-10-28

Source change 2026-10-28 (not a release gate): relational validation statistics, tokenizer catalogue, statistics UI and package split (commits `f881b6ac` .. `72f9dde8` and the documentation commit after them).

Implemented and exercised:
- Validation statistics live in relational tables (see [persistence](architecture/persistence.md)); the migration backfills older runs and was tested up and down on a database seeded at the previous head. A real validation of the six-record synthetic fixture dataset (full and 50% sample) produced per-record rows, aggregates and histograms that were read back over the API and shown in the browser.
- Tokenizer catalogue: `POST /api/tokenizers/check` loaded every one of the 18 catalogue entries (BERT, DistilBERT, RoBERTa, BioBERT, Bio_ClinicalBERT, BiomedBERT, SciBERT, DeBERTa, ALBERT, XLM-R, T5): 16 are compatible, and the two T5 entries are reported incompatible (no start token) and refused at dataset processing (a T5 processing job failed with that message). DeBERTa v3 was removed from the catalogue because it needs the `protobuf` package, which the environment does not have; DeBERTa large was checked in its place. Dataset processing, training (compact preset, 1 epoch), export, import into a separate Inference process and generation were run for distilbert (pad id 0) and roberta-base (pad id 1, start id 0).
- Backend split into `xreport_shared`, `xreport_studio`, `xreport_inference` and Angular split into `projects/{shared,studio,inference}`; each application was started on its own through its own entry point and serves only its own routes. `tests/e2e/test_studio_inference_exchange.py` starts both as separate processes and passed (about 2 minutes).
- Automated: 350 backend unit/integration tests (6 PostgreSQL tests skipped), 120 Angular unit tests in three projects, lint, and builds of both Angular applications.

Not verified or known limits:
- The `xreport-default` preset (six layers, 768 wide) failed while saving its first `.keras` checkpoint in this environment with an h5py `SystemError` from an in-memory file; the compact preset saved, exported and imported. The cause was not investigated and it was not compared with the previous commit.
- The launcher, `run_tests.bat`, the PyInstaller spec and the Tauri build were updated for the new entry points but only the unit assertions over them ran; no launcher menu action, desktop package or CUDA run was executed. PostgreSQL migrations, the Playwright UI end-to-end suite and the full `run_tests.bat` were not run.
- Dependency manifests are not split: one `pyproject.toml`/`uv.lock` and one `package.json` still serve both applications. Some Studio-oriented state types and API services (`types.ts`, `app-state.service`, training/inference API types) remain in the shared Angular project because the shared layout reaches them.
- Token counts per record exist in the schema but the validation wizard does not request a tokenizer yet, so token statistics are not produced.

Source change 2026-10-27 (not a release gate): Studio training pipeline audit (Keras 3.13.1 on torch 2.10 cu130), commits `af724c86`, `255610e7`, `680e80a4`.

Fixed defects, each reproduced first:
- The CPU choice trained on CUDA (Keras' torch backend defaults to CUDA). The worker now builds, loads and fits inside `keras.device`.
- Mixed precision failed at model build (fp16 input against fp32 BEiT weights). BEiT now runs under torch autocast, with `mixed_bfloat16` where supported.
- Dataloader workers > 0 crashed on Windows: the transform lambda could not be pickled.
- Keras' symbolic build fell back to an eager 83-image BEiT trace on the default CUDA device. The BEiT layer now declares its output shape and has a meta-device path. This was also what still broke the CPU choice.
- The torch compile option never took effect: `fit` ignored the wrapper. It was removed, together with the tf.data-era shuffle-buffer fields.
- Save checkpoints wrote a full ~770 MB file per epoch that nothing read. It now rewrites the checkpoint atomically each epoch. A stopped or failed run keeps its last completed epoch as a registered, resumable checkpoint; otherwise its folder is removed (unregistered `data/studio/checkpoints/XREPORT_20261027T172509` from an earlier run shows the old behaviour and was left in place).
- A duplicate custom checkpoint name overwrote an existing checkpoint; it now returns 409.

Gradient accumulation was added, using Keras' native optimizer support. The Studio source entry forces `KERAS_BACKEND=torch`.

Measurements: single runs, compact preset, 64 synthetic 1024 px images, batch 8, RTX 3060 Laptop 6 GB, steady-state epoch time.

| Configuration | Epoch time | Peak VRAM |
| --- | --- | --- |
| fp32, 0 workers (before) | 3.11 s | 3273 MB |
| fp32, 0 workers (after) | 2.67 s / 3.00 s | 2892 MB (after the shape fix) |
| fp32, 4 workers | 2.06 s | — |
| bf16, 4 workers | 1.92 s | — |
| batch 4 × accumulation 2, 4 workers | 2.92 s | 2070 MB |

The 0-worker after-change runs differ by about 12% between repeats, so the `verbose=0` change is not claimed as a speed-up. Each Windows loader worker re-imports torch, so the first batch of each loader starts several seconds later.

Validation:
- 329 backend unit/integration tests passed and 6 PostgreSQL cases were skipped. `test_packaging_boundary.py::test_xreport_import_closure_never_reaches_studio_modules` failed here and identically at the pre-change commit. The Model-page change had made the shared `encoder.py` import the pinned BEiT revision from the Studio-only `models.training.architecture`. Because the PyInstaller spec excludes that module, a packaged Inference build would have failed to import the encoder when loading an XREPORT checkpoint. The pin now lives in the shared `domain/checkpoint_exchange.py` (`ENCODER_REVISION`, next to `ENCODER_REPOSITORY`); after that fix the full unit/integration suites passed (330 passed, 6 skipped). No packaged build was rebuilt to confirm this.
- Ruff and Pyright passed on the changed modules. The Studio build and lint passed, and 113 Angular tests passed.

Live, against a source Studio on CUDA with the 6-row dataset:
- Mixed + 2 workers + accumulation 2 trained and registered a checkpoint.
- A duplicate name was refused with 409, and resume +1 epoch reached 3 epochs.
- A stop requested during epoch 2 registered the last completed epoch (2), and that checkpoint resumed. A stop before the first epoch left no folder.
- An invalid GPU index failed with a typed error and no folder. The CPU choice trained.
- VRAM returned to 0 MiB after each run.
- In the in-app browser, the wizard showed and sent gradient accumulation and had no compile controls. A 3-epoch bf16 run tracked the job API, and Stop during epoch 7 of a 50-epoch resume left the checkpoint at epoch 6.
- In the Inference Python environment, the mixed-precision, CPU and a pre-existing checkpoint each loaded and produced finite outputs on CPU and CUDA.

Not exercised: training the default preset (496.3M parameters, 410.6M trainable). Its baseline run filled the 6 GB GPU (5977 MiB, 100% utilisation) without finishing an epoch within minutes and was stopped; a failure after a saved epoch, live (covered by unit tests); import through a running XREPORT Inference server. QA working files are in `assets/QA/training-pipeline/`; the QA checkpoints were deleted.

Source change 2026-10-27 (not a release gate): Studio has a Model page for configuring, visualizing and saving XREPORT architectures (built-in XREPORT default, compact, balanced and large presets; custom presets in the new Studio table `model_architecture_presets`, head `8071b092e5f8`). Training now takes an `architecture_preset`, and checkpoints store the resolved architecture and the measured parameter counts. The shared transformer layers gained backward-compatible `ffn_units`, `dropout` and `key_dim` arguments. BEiT stays the only image encoder. Backend unit and integration tests (323 passed, 6 PostgreSQL skips), Ruff and Pyright on the changed modules, Studio build and lint, and 113 Angular unit tests passed. Live, against a source Studio on CUDA: the migration applied to the existing database; the Model page was checked in light/dark at 1440/1024/375 (live update, split-head validation, save, update, reload, JSON import, duplicate-name refusal, delete). A 1-epoch training from a custom preset built exactly the previewed parameter counts. The saved checkpoint and a fresh build from its configuration matched in counts, weight shapes and block configs, and the Inference Python environment loaded it. Resume (+1 epoch) and export also completed. Not exercised live: a checkpoint trained before this change (covered by unit tests) and import into a running XREPORT Inference server. Evidence is under `assets/QA/model-architecture/`. The Studio initial bundle budget warning (500 kB) predates this change and now reads 506 kB.

Source change 2026-10-27 (not a release gate): XREPORT Inference and XREPORT Studio now have fully separate database schemas. Each has its own ORM base, Alembic stream and version table (`xreport_inference_version` head `36d142391099`, `xreport_studio_version` head `9d4d3bce4ede`). Studio stores no inference settings (`inference: null` in its Settings API), and the dataset-record link was removed from inference reports. Each application rejects the other's database and the retired shared schema without modifying them. There is no conversion from the shared schema; the local development databases were recreated. Backend unit tests (305), Pyright on the changed modules, Angular build/lint for both projects, and 101 Angular unit tests passed. Against live source servers: Studio e2e passed 26/26; Inference e2e passed 21/26; 4 failures are pre-existing stale expectations (three browser cases expect 4 Settings tabs instead of the current 5, one API case expects a removed `features` settings field) and one font-request abort passed on rerun. Live checks then passed four flows. First, SQLite switching through the Studio Settings API: a new file was initialized, each database kept its own seed, and switching to the Inference database was refused with the previous configuration kept active. Second, both Settings → Database sections in the in-app browser (schema owner and revision, ownership error on the other app's file). Third, a real 1-epoch Studio training on synthetic images, then export, import into XREPORT Inference, and one deterministic generation saved to history. Fourth, a Studio restart, after which its checkpoint, datasets and settings persisted. PostgreSQL was not exercised: the user chose to skip it. The parametrized PostgreSQL contract (both schemas, cross-app refusal) is unrun. Evidence is under `assets/QA/database-separation/`.

Source change 2026-10-27 (not a release gate): the inference application is displayed as XREPORT Inference across UI, FastAPI title, desktop shell and launcher (identifiers, bundle ids and artifact names unchanged); checkpoint import now rejects a different encoder revision and inconsistent metadata, export enforces the import limits, unused dependencies were removed and Studio-only packages moved to the `studio` extra, and a Studio-side import boundary test was added. Backend unit tests (295) and Angular lint/build/unit tests passed; a live round trip then passed on isolated data roots with the pinned BEiT encoder: a small real Keras checkpoint (1 encoder/1 decoder layer, synthetic 19-token tokenizer, no training) was exported by the Studio API (199 MB), imported by XREPORT Inference through the real load and forward-pass smoke test with the tokenizer installed offline, listed as a ready custom model, and generated reports for a synthetic image in deterministic and detailed profiles. The text is untrained and proves technical interoperability only; real training, the encoder-bundled archive, conflict/cancel UI paths and packaged import remain unverified.

This is the canonical current-status summary for XREPORT. It records scoped
component confidence, active issues, and known limits; it is not a development
diary, detailed test report, release approval, or clinical-quality statement.

Release 3.3.1 (2026-10-22) is a launcher shutdown-wait patch over 3.3.0; its package scope is recorded in the campaign ledger. Release 3.3.0 (2026-10-22) supersedes the unpublished 3.2.1 candidate; it adds the legacy-compatibility cleanup, the self-hosted FastAPI SPA launcher lifecycle, and the portable-Node recovery fix. Its source smoke is recorded in the [validation campaign ledger](validation_campaign_ledger.md); the published-package scope is recorded there after publication.

The post-split `develop` source gate passed at `e8ad7bedd776800ee3099239d6b257ee4d5cd0c6` with hosted CI [40122131254](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/40122131254). At that SHA, XREPORT/Studio builds and lint passed, focused live browser suites passed 18/18 and 14/14, 54 checkpoint/job unit tests passed, Angular passed 79/79, and the built-server suite passed 9/9. The full Python collection contains 331 cases; only the focused suites were run locally. This source evidence does not upgrade release/package acceptance, model-dependent V05–V07, or excluded hardware scopes.

The published v3.2.0 release was validated at source
`a8d75a0c0d5f02a29c85e9ac58f4837fc3081449`. Before the post-split campaign, the
most recent application-source Tier 0 validation targeted
`b98a9b60492b63fb41a904e3cff4815325b8d075`. On
2026-10-21 the application source dropped legacy compatibility layers: the
Alembic stream was squashed to the current-schema baseline `b1e7a2c4d908`,
database-configuration bootstrap now creates a default SQLite configuration
instead of importing legacy `.env`/`DATABASE_*` values, the former
settings-JSON cleanup was removed, and the client no longer migrates a
browser-stored theme preference into the database. A later working-tree change
adds migration `42c8e1f7b0d3` to move inference process policy to `.env` and
adds environment-driven CORS configuration. The scoped follow-up was validated
on `ef1ae34ad8e9dc073415e5ec1a0eaec588f81999`: rendered settings UI and
isolated SQLite persistence, supported environment parsing/CORS, and migration
from `b1e7a2c4d908` to `42c8e1f7b0d3` passed. Post-change Windows reruns
passed the backend suite (excluding live-server E2E), all client unit tests,
both lint gates, and the browser/API E2E suites except the `S23` case that
requires the still-absent `s27` fixture pack. On 2026-10-21, the current
working-tree launcher started on standard ports `5003`/`8003` with isolated
resources; Inference, Reports, Dataset, Training, and Settings loaded. Settings
seed `317` survived navigation and launcher restart; backend health reported
version `3.2.1`. SQLite integrity passed, and Alembic reached `42c8e1f7b0d3`.
Historically, the v3.2.1 candidate restored migration compatibility from published revisions
`f48a7c2e91b6` and `e91a4f6c2d73`, including the prior browser theme
preference. Sixteen focused database-initialization tests, four version/OpenAPI
tests, 65 client unit tests, and both lint gates passed. Candidate commit
`1a15d24cb07fecbf66601940752a164c47972fc7` passed hosted CI run
[39667709442](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/39667709442),
including backend unit tests, Ruff, Pyright, PostgreSQL persistence, retained API
E2E, client build/server, lint, and unit tests. Official packages remain open;
Tier 0 remains `PARTIAL` until that gate finishes. Follow-on cleanup removed the neutral validation
hash field, historical cache-directory sweep, and npm peer-dependency bypass,
and replaced the API type generator. A clean `npm ci`, dependency-tree check,
API type generation, client build/lint, 64 client unit tests, 18 dedicated
browser E2E tests, backend Ruff, and backend Pyright passed. The standard runner
reported 245 passed, 3 skipped, and 4 failed; the S23 failure lacks its fixture
pack, the upload-size case passed on retry, the S10 retry could not connect to
the runner listener, and the request-gate test failed before its body due to
Windows asyncio loop initialization. These end-to-end scopes remain `PARTIAL`.
SonarCloud is excluded from project gates
because the available plan does not support analysis for this repository's
over-50k LOC. The
[validation campaign ledger](validation_campaign_ledger.md) records the exact
evidence and remaining boundaries.

## Maintenance rules

- Keep this ledger focused on current status and active limits. Remove resolved
  findings from the issue list once their lasting decision is recorded in the
  relevant architecture or workflow documentation.
- Use `VALIDATED` only for the stated scope and evidence boundary. Tests,
  source inspection, browser observation, provider access, package acceptance,
  and clinical review prove different things.
- Update the component row and relevant topic document when an implementation,
  validation result, blocker, or important limitation changes.
- Keep architecture and workflow documents authoritative for contracts. Use
  the campaign ledger for gate status and release scope; do not duplicate run
  histories here.
- Store working captures under `assets/QA/`. Track only artifacts required by
  project tooling; do not add standalone per-slice summaries or duplicate
  receipts.

## Status meanings

| Status | Meaning |
| --- | --- |
| `VALIDATED` | Confirmed through meaningful evidence for the stated scope. |
| `WORKING` | Implemented and partly exercised, but not fully validated. |
| `PARTIAL` | Confirmed only for part of the expected behavior or environment. |
| `BROKEN` | Known not to work correctly for the stated scope. |
| `BLOCKED` | Validation depends on unavailable access, hardware, service, or another external condition. |
| `UNVALIDATED` | Evidence is insufficient to claim that the implementation works. |
| `NOT_IMPLEMENTED` | The capability is absent. |
| `DEPRECATED` | Intentionally retired or scheduled for removal. |

README capture check 2026-10-27: XREPORT Inference launched with an isolated resource directory. LaunchStudio twice stopped at the Studio environment probe after dependency synchronization; the launcher reported only a truncated Python traceback, so the cause remains unconfirmed. The Studio source entry started with KERAS_BACKEND=torch and served the empty dataset page, which does not validate the launcher. The launcher recreated the Studio Standard environment during this check.

Launcher fix 2026-10-27: the root cause of the LaunchStudio probe failure above is confirmed. Windows PowerShell 5.1 removed the embedded double quotes from the inline Python probe, so the probe raised `NameError: name 'cuda' is not defined`. KERAS_BACKEND=torch did reach the probe through `settings/.env`. With `$ErrorActionPreference = 'Stop'`, the traceback's first stderr line became a terminating NativeCommandError that hid the rest of the traceback. A failed probe also removes `runtimes/studio-install-state.json`, so the next launch recreates `.venv-studio`; without `-StudioProfile`, that recreation uses Standard. This explains the earlier Standard recreation. The probe now uses single-quoted Python, sets KERAS_BACKEND=torch itself, runs with `Continue`, returns false on failure, and prints the full output. Validation used the real `-Action LaunchStudio` on the existing CUDA `.venv-studio` (torch 2.10.0+cu130), which was not recreated: the probe passed without reinstalling, the API and Studio UI were served on port 5013, and `-Action KillProcesses -App Studio` stopped it. Ports 5013 and 4213 were then free. `test_launcher_studio.py` passed 14/14 with one new regression case. The Standard profile was not exercised. Evidence is in `assets/QA/studio-launcher-probe-2026-10-27/`.

Source check 2026-10-27 (working tree): the retired database-revision bridge was removed and six focused SQLite migration and rejection cases passed. The full Windows runner reported 317 passed, 3 skipped, and 4 failed in the XREPORT Python phase; Studio reported 15 passed and 11 failed, including ten preview connection refusals and one unavailable tokenizer; Angular passed 100 tests. Two XREPORT browser cases expected four tabs but found five, one API case expected a missing settings field, and the lifespan case encountered the live runner's resource lock. PostgreSQL was not configured. These results do not change the historical release gates. Details are in the ignored `assets/QA/shim-cleanup-full-tests.log`.

Source-validated 2026-10-26 (not part of a release gate): the shared client gained a full-width system status bar (`GET /api/system/status`), a notification centre with toasts, and a structured generation-metadata view. Backend unit tests, Angular lint/build/unit tests, and a manual light/dark and backend-offline check on XREPORT passed. The Studio profile shares the same layout and endpoint and was built but not exercised in a browser. Notifications for inference outcomes are emitted only while the inference page is mounted.

## Component status

Statuses refer to the most recent evidence recorded for each scope, not to an
implicit fresh run on every checkout. Tests and contracts linked below identify
where to revalidate after a change.

### Runtime, configuration, and security

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `runtime.source.startup` | `PARTIAL` | Normal Windows Launch was exercised with isolated resources: FastAPI served the rendered SPA, built assets, direct `/reports` fallback, and same-origin `/api` responses; Ctrl+C called the authenticated shutdown endpoint and left the owned backend process stopped, with ports 5003/8003 clear. An unidentified test listener on `FASTAPI_PORT` caused a fail-closed abort and remained running during that check. Focused source-hosting tests passed 7/7 and the launcher parsed cleanly. The frontend state manifest now fingerprints Node/npm and verifies every generated browser file; focused launcher/configuration tests passed 52 cases, and a temporary fixture smoke passed unchanged reuse, missing-output invalidation, npm-version invalidation, and per-app state checks. The launcher reports aligned millisecond timings for its startup phases, including total time to backend health readiness, and suppresses live health progress during foreground FastAPI output; this timing change received a syntax parse only, with launch behavior not rerun. Tauri release configs still declare `frontendDist: "ui"` and the Rust sidecar stop path is present; Tauri dev was not exercised because desktop Node dependencies/debug binary and MSVC `cl.exe` were unavailable. Existing manifest, migration, CI, and earlier rendered-flow evidence remain scoped as previously recorded; package validation remains open. | [Startup](runtime/startup.md), [campaign](validation_campaign_ledger.md) |
| `runtime.desktop.packaged` | `PARTIAL` | Published v3.2.0 CPU/CUDA portable and MSI packages passed the scoped acceptance described in the campaign ledger. CUDA fallback without an NVIDIA GPU and cross-variant performance remain untested. | [Campaign](validation_campaign_ledger.md), [deployment](runtime/deployment.md) |
| `runtime.containerized` | `NOT_IMPLEMENTED` | No supported container image or deployment contract exists. | [Runtime modes](runtime/modes.md) |
| `configuration.runtime_settings` | `VALIDATED` | On `ef1ae34ad8e9dc073415e5ec1a0eaec588f81999`, rendered settings tabs passed autosave, bounds, reset, reload/restart persistence, per-database isolation, failed-connection draft handling, and the Settings API policy boundary. Supported `.env` parsing and CORS behavior passed isolated fixtures. The ignored local `settings/.env` was not changed during the settings checks. | [Configuration](runtime/configuration.md), [campaign](validation_campaign_ledger.md), [environment tests](../../app/tests/unit/test_runtime_startup.py), [settings E2E](../../app/tests/e2e/test_angular_ui.py) |
| `configuration.database_runtime` | `PARTIAL` | Database values remain uncommitted drafts until the current candidate passes a connection test and receives explicit confirmation; SQLite settings isolation, restart persistence, failed PostgreSQL handling, rollback, and the request-drain gate all passed. A successful authenticated PostgreSQL connection and live switch remain unverified; the user explicitly deferred PostgreSQL testing on 2026-10-21, so no connection or switch was attempted in this run. | [Configuration](runtime/configuration.md), [Database configuration tests](../../app/tests/unit/test_settings_persistence.py) |
| `security.authentication` | `NOT_IMPLEMENTED` | The current boundary is trusted local use; external exposure is not covered. | [Execution and data flow](architecture/execution_and_data_flow.md) |

### Application split and checkpoint exchange

PR #9 review fixes passed 272 backend unit tests, Ruff and Pyright locally, and hosted [CI run 39931654917](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/39931654917) at `0121c6b4`, including both client builds and both profiles' retained API E2E. SonarCloud and GitGuardian passed. Source migration selection and checkpoint export concurrency findings are resolved; packaged import, real-data migration and a full Studio UI training journey remain unverified.

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `app.profiles` | `WORKING` | Merged on `develop` by PR #9: `create_app(profile)` mounts only profile routers; route sets were verified on live servers (XREPORT: inference, jobs, settings, exchange; Studio: upload, preparation, training, validation, jobs, settings, exchange) and by subprocess tests. The XREPORT static import closure is free of Studio modules. Both applications ran side by side on ports 5003/5013 with isolated data. | [Applications](architecture/applications_and_checkpoint_exchange.md) |
| `app.studio.frontend` | `WORKING` | `studio-angular` builds separately; the XREPORT bundle contains no dataset/training chunks or `/api/training|preparation|upload|validation` strings (grep of the production output). V11 built and linted both applications; 79 Angular unit tests passed on `e8ad7bedd776800ee3099239d6b257ee4d5cd0c6`. Rendered browser checks covered Studio navigation, dataset and training pages. | [Applications](architecture/applications_and_checkpoint_exchange.md) |
| `checkpoint.exchange` | `PARTIAL` | Unit coverage confirms the format-2 round trip, idempotency, conflicts, cancellation and rollback, and rejection of legacy/manifestless archives; the earlier live round trip used format 1. V08 UI on `9b298c54` rejected a lone `.keras` and a manifestless four-core ZIP before job admission. V11 adds `app/tests/e2e/test_angular_ui.py::test_checkpoint_import_rejects_manifestless_archive_in_live_ui`, which exercises the live XREPORT import dialog and verifies the typed missing-manifest error with no import progress/job. V05 Studio training is blocked by the missing pinned encoder, so positive v2 export/import, custom-model inference/restart, checksum/version/conflict UI cases, import cancellation/recovery and packaged import remain unverified. | [Applications](architecture/applications_and_checkpoint_exchange.md), [campaign](validation_campaign_ledger.md) |
| `runtime.studio.launcher` | `PARTIAL` | Studio now has an isolated profiled environment, first-launch setup, profile reuse/switching, retryable setup state, a dedicated install action, and uninstall cleanup. Eleven focused launcher tests, PowerShell parsing, `uv lock --check`, the XREPORT CUDA-only dry-run, and both Studio profile dry-runs pass. The new first-launch installation and profile replacement were not exercised against real dependency downloads; earlier Windows launch, build freshness, process scoping, and port-release evidence remains applicable to those behaviors. | [Startup](runtime/startup.md) |
| `desktop.packaging.inference_only` | `UNVALIDATED` | PyInstaller spec and bundle audit exclude Studio modules/assets; no CPU/CUDA package was built, so package contents and import into a packaged build were not observed. | [Deployment](runtime/deployment.md) |

### Backend and persistence

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `backend.api.contracts`, `backend.jobs.lifecycle`, `backend.ml_import_boundaries` | `VALIDATED` | Covered API contracts, generic job lifecycle, and lightweight-import boundaries are validated; this does not cover every endpoint or heavy workflow. | [Backend API](architecture/backend_api.md), [API tests](../../app/tests/e2e/test_api_inference_settings.py), [job tests](../../app/tests/integration/test_job_lifecycle_api.py), [import test](../../app/tests/unit/test_runtime_startup.py) |
| `persistence.sqlite_migrations`, `persistence.inference_history`, `persistence.postgresql`, `persistence.checkpoint_registry` | `PARTIAL` | Since 2026-10-27 each application has its own stream (Inference `36d142391099`, Studio `8071b092e5f8` since the architecture-preset migration on top of `9d4d3bce4ede`); SQLite initialization, ownership refusal, drift, rollback and unknown-revision cases pass for both schemas. PostgreSQL for the separated schemas is unverified. Earlier history: the shared stream began at `b1e7a2c4d908`; published revisions `f48a7c2e91b6` and `e91a4f6c2d73` are rejected and require database recreation. Focused SQLite tests passed for fresh/current migration, removed revisions, schema rejection, and rollback. Hosted run `39667709442` passed the earlier PostgreSQL persistence contract; PostgreSQL has not been revalidated after this cleanup. Crash recovery and every production configuration remain outside scope. | [Persistence](architecture/persistence.md), [SQLite tests](../../app/tests/unit/test_settings_persistence.py), [PostgreSQL contract test](../../app/tests/integration/test_persistence_contract.py) |

### User workflows

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `workflow.dataset_upload_and_preparation` | `VALIDATED` | Synthetic upload, matching, confirmation, processing, and deletion protections are exercised. Representative-data, large-scale, and packaged workflows are not established. | [Workflows](operations/workflows.md), [dataset E2E](../../app/tests/e2e/test_dataset_workflow.py) |
| `workflow.training_and_resume` | `VALIDATED` | Minimal local training, cancellation, resume, and checkpoint persistence are exercised with technical fixtures. Model quality and representative training performance are not established. | [Workflows](operations/workflows.md), [training E2E](../../app/tests/e2e/test_training_api.py) |
| `workflow.dataset_validation`, `workflow.checkpoint_evaluation` | `VALIDATED` | Synthetic technical validation and checkpoint evaluation complete and persist reports. No clinical or representative-data quality claim follows. | [Workflows](operations/workflows.md), [validation tests](../../app/tests/unit/test_evaluation_validation.py) |
| `workflow.inference.generation_pipeline` | `VALIDATED` | The end-to-end contract is exercised with pinned local models and synthetic/de-identified inputs; provider-wide quality and all failure/reuse conditions remain outside. | [Local inference models](runtime/local_inference_models.md), [inference E2E](../../app/tests/e2e/test_api_inference_settings.py) |

### Model providers

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `inference.catalogue` | `VALIDATED` | Public/custom separation, pinned entries, readiness, access, and declared output sections are covered. Catalogue size is not a product invariant. | [Local inference models](runtime/local_inference_models.md) |
| `inference.model.cxrmate-multi` | `PARTIAL` | On 2026-10-21, one live local image-to-report run completed on the pinned revision `330721b9aa5bba201a3eb88eba4dd9a6607f3e7a` using a public CC0 chest radiograph ([source](https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg)); runtime provenance resolved to `cuda:0` with CUDA used. The catalogue reports `ready`, while `validation_status` remains `pending` and the validation receipt is missing. This proves technical execution only, with no model-quality or clinical claim. Screenshot: [live inference](../QA/codex-live-inference-20261021.png). | [Local inference models](runtime/local_inference_models.md), [campaign](validation_campaign_ledger.md) |
| `inference.model.cxrmate-ed` | `DEPRECATED` | Retired after three distinct fixtures collapsed to the same full report. Historical user data and model files remain preserved. | [Local inference models](runtime/local_inference_models.md) |
| `inference.model.chexone`, `inference.model.cxrmate-2` | `VALIDATED` | Pinned technical generation and conservative output review passed for bounded fixtures; diagnostic accuracy is not established. | [Local inference models](runtime/local_inference_models.md) |
| `inference.model.medgemma` | `BLOCKED` | Gated-provider behavior is validated, but real installation and inference require authorized access. | [Gated access](runtime/local_inference_models.md); `ISSUE-002` below |

### User interface and test infrastructure

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `ui.settings`, `ui.inference.catalogue_and_sections`, `ui.reports.history`, `ui.startup_gate`, `ui.shell.routing_theme_guidance`, `ui.validation.report_review` | `VALIDATED` | Settings tabs and controls are covered by focused client/API tests and browser checks using technical fixtures: non-database changes autosave and restore after reload, while database edits remain drafts until a successful current-candidate connection test and explicit confirmation; failed probes leave the active backend unchanged. Persisted report detail edits prompt before internal navigation, including browser Back, protect reload and close where supported, block navigation during saves and deletes, and retain drafts after failed requests. Focused unit and browser coverage passed. The Data access setting and its application-level filesystem gate were removed from both profiles (migration `7d3f9a1c5e82`); Reports filters auto-apply, theme changes are silent, report detail highlights edited words in the original output, and Storage settings use a block layout. These were exercised in the in-app browser (light and dark) and client/unit tests; the Python e2e files were edited but not run, and `test_server_app_import_does_not_load_heavy_runtime_modules` (route count 31 vs 30) already failed before this change. This is not full product accessibility certification. | [UI experience](ui/experience.md), [browser E2E](../../app/tests/e2e/test_angular_ui.py) |
| `ui.inference.workflow` | `PARTIAL` | Completed inference now links its editable draft to saved history. Save patches only changed canonical report rows, including the single persisted row for study-level output. The 79-test Angular suite, focused backend coverage, history API E2E, and edit/save/reload browser journey passed locally; the browser journey used a deterministic job fixture and real SQLite history persistence. Real-provider inference and broader provider-contract coverage remain open. | [Local inference models](runtime/local_inference_models.md), [UI experience](ui/experience.md), [draft persistence specs](../../app/client/src/app/pages/inference-draft-persistence.page.spec.ts), [browser E2E](../../app/tests/e2e/test_angular_ui.py) 2026-10-26: inference tracking now survives navigation (job id in shared state; running and completed jobs restore on return) and a Run-on control adds per-image batch runs; both were exercised live with CXRMate Multi TF (return mid-run and after completion, 3-image batch), plus Angular and unit regressions. Settings → Storage validates and moves the storage root (applied on restart); the copy path is unit-tested and the live UI was only exercised through Check folder, not a 20 GB move. 2026-10-26: the generation profile stays editable after a draft, Run-on options carry descriptions (stale unsupported study scope falls back to per-image), and per-image runs get a report selector with independent save; the 88-test Angular suite passed and all three modes were exercised live with CXRMate Multi TF on synthetic images (4 rows per-image, 1 row study, 1 row selected; profile change plus regenerate; per-report edit/save). Clinical-image quality was not assessed. 2026-10-26: model memory management and per-model studies. One resident model is kept across providers and released by an idle timeout (`inference.model_idle_timeout`, migration `c3b8e5d1a7f4`) or by the status-bar **Free memory** button (`POST /api/inference/runtime/release`, 409 during runs). XREPORT checkpoints are now reused across runs and per-image batches. A live probe on an RTX 3060 6 GB with synthetic images measured CheXOne at 5.36 GB VRAM, falling to 1.05 GB after each manual release (baseline 0.99 GB), with no growth across 3 CXRMate Multi TF ↔ CheXOne switch cycles. A release during a run returned 409 and the run completed; a 5 s idle timeout did not interrupt a run; a 20 s idle timeout released the model about 25 s after the run (`assets/QA/memory-management/memory-probe.log`). In the in-app browser, the CXRMate study (2 images, image 2 selected, generated draft with an unsaved edit) survived a switch to CheXOne, a Reports round trip and Free memory without prompts. Clear study reset only the active model. Angular passed 100/100 and focused backend suites passed 141. XREPORT-checkpoint residency is unit-tested only because no checkpoint is registered locally. |
| `ui.dataset_and_training_surfaces` | `PARTIAL` | V00–V04 browser checks on the post-split `develop` source used synthetic SQLite data: both profiles rendered, Studio imported clean/malformed/missing-text/partial CSVs, the viewer navigated image rows, dataset processing produced 7 train/1 validation rows, and pixel/text/image statistics persisted across reload. V09 completed distinct jobs and app-scoped restart checks, with cancellation and duplicate-start limits retained. V10 reviewed both apps across four viewport sizes; the Studio dataset table clips the `Rows` column at 320px (`ISSUE-005`). V05 CPU training is blocked because the pinned BEiT encoder is absent. Packaged workflows remain open; no representative or clinical data was used. The processing split field accepts up to 1.0 while the API caps it at 0.5 (`ISSUE-003`). | [UI experience](ui/experience.md), [campaign](validation_campaign_ledger.md), [dataset workflow E2E](../../app/tests/e2e/test_dataset_workflow.py) |
| `test.infrastructure.suite_consolidation` | `VALIDATED` | Exact SHA `e8ad7bedd776800ee3099239d6b257ee4d5cd0c6` has 41 test source files and 419 cases: 26 Python files/331 collected cases, 14 Angular specs/79 passed, and 1 Node file/9 passed. V11 ran the live XREPORT browser suite (18/18), Studio UI/dataset/filesystem suites (14/14), 54 checkpoint/job unit tests, both client builds, and both lint targets; hosted CI 40122131254 passed. The full Python suite was collected but not executed locally; focused local receipts are under `assets/QA/validation_campaign/xr-20261024-01/`. | [Testing and quality](coding/testing_and_quality.md), [Python suites](../../app/tests/unit/test_runtime_startup.py), [E2E suites](../../app/tests/e2e/test_angular_ui.py), [campaign](validation_campaign_ledger.md) |
| `test.infrastructure.windows_cache` | `PARTIAL` | Runner and direct pytest instructions use canonical `runtimes/cache` paths. Launcher cache clear and uninstall now target only current runtime/build locations; PowerShell parsing passed, but these two actions were not exercised in this run. Historical cache locations are no longer enumerated or removed. | [Testing and quality](coding/testing_and_quality.md), [startup](runtime/startup.md), [campaign](validation_campaign_ledger.md) |

## Active issues

| ID | Component | Severity | State | Current impact and next evidence |
| --- | --- | --- | --- | --- |
| `ISSUE-002` | `inference.model.medgemma` | `MEDIUM` | `BLOCKED` | The gated model cannot be installed without authorized provider access. After access is configured, install the exact pinned revision and capture real inference before changing status. |
| `ISSUE-003` | `ui.dataset.processing.validation_split` | `MEDIUM` | `OPEN` | The Studio field advertises `0–1` (`max=1`) but the backend accepts only `0.05–0.5`; entering `1.2` reaches the API and displays a typed 422. Proposed remedy: set the field maximum and Angular validation to `0.5`, then verify both boundary values. No production fix was applied during this validation campaign. |
| `ISSUE-005` | `ui.dataset.table.responsive_rows` | `LOW` | `OPEN` | At 320px viewport width, the Studio dataset table's `Rows` heading and values occupy x=302–336 and are clipped by the table container while the document itself stays 320px wide. Proposed remedy: reflow or horizontally scroll the table inside its container while keeping row counts reachable. No production UI change was applied. |

## Evidence ownership

- Architecture and topic documents own current structures, contracts, and
  runtime behavior.
- This ledger owns component status and active issues. The
  [validation campaign ledger](validation_campaign_ledger.md) owns the concise
  gate map and release scope.
- Tests and CI runs are evidence for their stated scope, not a substitute for
  observing the browser, hardware, provider, or package boundary being claimed.
- Store QA working files under `assets/QA/`; the approved release manifest is
  tracked because the release verifier consumes it. Do not preserve duplicate
  summaries, raw receipts, or resolved-issue history as parallel records.
