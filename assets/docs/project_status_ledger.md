# XREPORT Project Status Ledger

Last updated: 2026-11-25

This is the canonical current-status summary for XREPORT. It records scoped
component confidence, active issues, and known limits; it is not a development
diary, detailed test report, release approval, or clinical-quality statement.
Run evidence lives in the [validation campaign ledger](validation_campaign_ledger.md)
and release contents in the [changelog](../../CHANGELOG.md).

## Current state

- Release line: 4.0.1 (patch over 4.0.0, the Inference/Studio split). Only
  XREPORT Inference is packaged; Studio runs from source. The 4.0.1 gate is
  recorded in the campaign ledger.
- Source gate at 4.0.1: Ruff and Pyright clean; 355 backend unit tests (2 skipped); 122
  Angular unit tests; both builds, lint and the built-server test; live API
  E2E (Inference 7, Studio 11) and Playwright browser suites (Inference 19,
  Studio 16) on isolated SQLite roots. Hosted CI on `develop` had been failing
  since 2026-11-09 on one stale launcher-cleanup assertion; fixed in 4.0.1.
- Validation debt that still applies: PostgreSQL for the separated schemas,
  Studio training of the default preset on the 6 GB reference GPU (it filled
  the GPU without finishing an epoch), packaged inference with a real model,
  packaged checkpoint import, GPU-less CUDA fallback, real (non-dry-run)
  launcher cleanup and MSI uninstall through `setup_and_maintenance.ps1`,
  screen-reader review, and the full `run_tests.bat` workflow (its Studio
  browser phase races the preview server).
- Known limits: old per-version runtime extractions under `%LOCALAPPDATA%` are not pruned on upgrade (about 0.9 GB per CPU version); one `pyproject.toml`/`uv.lock` and one `package.json` serve
  both applications; token statistics do not set `truncated` or
  `max_report_size`; unknown `/api/*` paths in the packaged build return the
  Angular index with HTTP 200; model output quality is not clinically
  validated. A locally modified model snapshot is rejected by the integrity
  check until it is re-downloaded (observed for CXRMate Multi TF).

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
- Store temporary captures and receipts under ignored `runtimes/cache/` and
  remove them after the run. Keep durable outcomes in the existing ledgers;
  retain only source fixtures and machine-required release records.

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

Launcher fix 2026-10-27: the root cause of the LaunchStudio probe failure above is confirmed. Windows PowerShell 5.1 removed the embedded double quotes from the inline Python probe, so the probe raised `NameError: name 'cuda' is not defined`. KERAS_BACKEND=torch did reach the probe through `settings/.env`. With `$ErrorActionPreference = 'Stop'`, the traceback's first stderr line became a terminating NativeCommandError that hid the rest of the traceback. A failed probe also removes `runtimes/studio-install-state.json`, so the next launch recreates `.venv-studio`; without `-StudioProfile`, that recreation uses Standard. This explains the earlier Standard recreation. The probe now uses single-quoted Python, sets KERAS_BACKEND=torch itself, runs with `Continue`, returns false on failure, and prints the full output. Validation used the real `-Action LaunchStudio` on the existing CUDA `.venv-studio` (torch 2.10.0+cu130), which was not recreated: the probe passed without reinstalling, the API and Studio UI were served on port 5013, and `-Action KillProcesses -App Studio` stopped it. Ports 5013 and 4213 were then free. `test_launcher_studio.py` passed 14/14 with one new regression case. The Standard profile was not exercised.

Source check 2026-10-27 (working tree): the retired database-revision bridge was removed and six focused SQLite migration and rejection cases passed. The full Windows runner reported 317 passed, 3 skipped, and 4 failed in the XREPORT Python phase; Studio reported 15 passed and 11 failed, including ten preview connection refusals and one unavailable tokenizer; Angular passed 100 tests. Two XREPORT browser cases expected four tabs but found five, one API case expected a missing settings field, and the lifespan case encountered the live runner's resource lock. PostgreSQL was not configured. These results do not change the historical release gates.

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
| `desktop.packaging.inference_only` | `VALIDATED` | Release 4.0.1 CPU and CUDA packages built by CI from the tag passed artifact verification, payload inspection (no Studio module, secrets, databases or logs) and packaged startup smoke locally. The CPU MSI installed (4.0.0), upgraded in place to 4.0.1 with user settings preserved, and uninstalled keeping user data. Old per-version runtime extractions are not pruned on upgrade. Packaged inference with a model, packaged checkpoint import, the CUDA MSI lifecycle and GPU-less CUDA fallback were not exercised. | [Deployment](runtime/deployment.md), [campaign](validation_campaign_ledger.md) |

### Backend and persistence

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `backend.api.contracts`, `backend.jobs.lifecycle`, `backend.ml_import_boundaries` | `VALIDATED` | Covered API contracts, generic job lifecycle, and lightweight-import boundaries are validated; this does not cover every endpoint or heavy workflow. | [Backend API](architecture/backend_api.md), [API tests](../../app/tests/e2e/test_api_inference_settings.py), [job tests](../../app/tests/integration/test_job_lifecycle_api.py), [import test](../../app/tests/unit/test_runtime_startup.py) |
| `persistence.sqlite_migrations`, `persistence.inference_history`, `persistence.postgresql`, `persistence.checkpoint_registry` | `PARTIAL` | Since 2026-10-27 each application has its own stream (Inference `36d142391099`, Studio `8071b092e5f8` since the architecture-preset migration on top of `9d4d3bce4ede`); SQLite initialization, ownership refusal, drift, rollback and unknown-revision cases pass for both schemas. PostgreSQL for the separated schemas is unverified. Earlier history: the shared stream began at `b1e7a2c4d908`; published revisions `f48a7c2e91b6` and `e91a4f6c2d73` are rejected and require database recreation. Focused SQLite tests passed for fresh/current migration, removed revisions, schema rejection, and rollback. Hosted run `39667709442` passed the earlier PostgreSQL persistence contract; PostgreSQL has not been revalidated after this cleanup. Crash recovery and every production configuration remain outside scope. | [Persistence](architecture/persistence.md), [SQLite tests](../../app/tests/unit/test_settings_persistence.py), [PostgreSQL contract test](../../app/tests/integration/test_persistence_contract.py) |

### User workflows

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `workflow.dataset_upload_and_preparation` | `VALIDATED` | Synthetic upload, matching, confirmation, processing, and deletion protections passed in prior workflow evidence. In the 2026-11-11 hidden in-app-browser audit, the synthetic CSV parsed as 8 rows × 2 columns but the folder picker did not surface; this mode is unverified until retested in the normal foreground launcher. Representative-data, large-scale, and packaged workflows are not established. | [Workflows](operations/workflows.md), [dataset E2E](../../app/tests/e2e/test_dataset_workflow.py) |
| `workflow.training_and_resume` | `VALIDATED` | Minimal local training, cancellation, resume, and checkpoint persistence are exercised with technical fixtures. Model quality and representative training performance are not established. | [Workflows](operations/workflows.md), [training E2E](../../app/tests/e2e/test_training_api.py) |
| `workflow.dataset_validation`, `workflow.checkpoint_evaluation` | `VALIDATED` | Synthetic technical validation and checkpoint evaluation complete and persist reports. No clinical or representative-data quality claim follows. | [Workflows](operations/workflows.md), [validation tests](../../app/tests/unit/test_evaluation_validation.py) |
| `workflow.inference.generation_pipeline` | `VALIDATED` | The end-to-end contract is exercised with pinned local models and synthetic/de-identified inputs; provider-wide quality and all failure/reuse conditions remain outside. | [Local inference models](runtime/local_inference_models.md), [inference E2E](../../app/tests/e2e/test_api_inference_settings.py) |

### Model providers

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `inference.catalogue` | `VALIDATED` | Public/custom separation, pinned entries, readiness, access, and declared output sections are covered. Catalogue size is not a product invariant. | [Local inference models](runtime/local_inference_models.md) |
| `inference.model.cxrmate-multi` | `PARTIAL` | On 2026-10-21, one live local image-to-report run completed on the pinned revision `330721b9aa5bba201a3eb88eba4dd9a6607f3e7a` using a public CC0 chest radiograph ([source](https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg)); runtime provenance resolved to `cuda:0` with CUDA used. The catalogue reports `ready`, while `validation_status` remains `pending` and the validation receipt is missing. This proves technical execution only, with no model-quality or clinical claim. | [Local inference models](runtime/local_inference_models.md), [campaign](validation_campaign_ledger.md) |
| `inference.model.cxrmate-ed` | `DEPRECATED` | Retired after three distinct fixtures collapsed to the same full report. Historical user data and model files remain preserved. | [Local inference models](runtime/local_inference_models.md) |
| `inference.model.chexone`, `inference.model.cxrmate-2` | `VALIDATED` | Pinned technical generation and conservative output review passed for bounded fixtures; diagnostic accuracy is not established. | [Local inference models](runtime/local_inference_models.md) |
| `inference.model.medgemma` | `BLOCKED` | Gated-provider behavior is validated, but real installation and inference require authorized access. | [Gated access](runtime/local_inference_models.md); `ISSUE-002` below |

### User interface and test infrastructure

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `ui.settings`, `ui.inference.catalogue_and_sections`, `ui.reports.history`, `ui.startup_gate`, `ui.shell.routing_theme_guidance`, `ui.validation.report_review` | `VALIDATED` | Settings tabs and controls are covered by focused client/API tests and browser checks using technical fixtures: non-database changes autosave and restore after reload, while database edits remain drafts until a successful current-candidate connection test and explicit confirmation; failed probes leave the active backend unchanged. Persisted report detail edits prompt before internal navigation, including browser Back, protect reload and close where supported, block navigation during saves and deletes, and retain drafts after failed requests. Focused unit and browser coverage passed. The Data access setting and its application-level filesystem gate were removed from both profiles (migration `7d3f9a1c5e82`); Reports filters auto-apply, theme changes are silent, report detail highlights edited words in the original output, and Storage settings use a block layout. These were exercised in the in-app browser (light and dark) and client/unit tests; the Python e2e files were edited but not run, and `test_server_app_import_does_not_load_heavy_runtime_modules` (route count 31 vs 30) already failed before this change. This is not full product accessibility certification. | [UI experience](ui/experience.md), [browser E2E](../../app/tests/e2e/test_angular_ui.py) |
| `ui.inference.workflow` | `PARTIAL` | Completed inference now links its editable draft to saved history. Save patches only changed canonical report rows, including the single persisted row for study-level output. The 79-test Angular suite, focused backend coverage, history API E2E, and edit/save/reload browser journey passed locally; the browser journey used a deterministic job fixture and real SQLite history persistence. Real-provider inference and broader provider-contract coverage remain open. | [Local inference models](runtime/local_inference_models.md), [UI experience](ui/experience.md), [draft persistence specs](../../app/client/projects/inference/src/app/pages/inference-draft-persistence.page.spec.ts), [browser E2E](../../app/tests/e2e/test_angular_ui.py) 2026-10-26: inference tracking now survives navigation (job id in shared state; running and completed jobs restore on return) and a Run-on control adds per-image batch runs; both were exercised live with CXRMate Multi TF (return mid-run and after completion, 3-image batch), plus Angular and unit regressions. Settings → Storage validates and moves the storage root (applied on restart); the copy path is unit-tested and the live UI was only exercised through Check folder, not a 20 GB move. 2026-10-26: the generation profile stays editable after a draft, Run-on options carry descriptions (stale unsupported study scope falls back to per-image), and per-image runs get a report selector with independent save; the 88-test Angular suite passed and all three modes were exercised live with CXRMate Multi TF on synthetic images (4 rows per-image, 1 row study, 1 row selected; profile change plus regenerate; per-report edit/save). Clinical-image quality was not assessed. 2026-10-26: model memory management and per-model studies. One resident model is kept across providers and released by an idle timeout (`inference.model_idle_timeout`, migration `c3b8e5d1a7f4`) or by the status-bar **Free memory** button (`POST /api/inference/runtime/release`, 409 during runs). XREPORT checkpoints are now reused across runs and per-image batches. A live probe on an RTX 3060 6 GB with synthetic images measured CheXOne at 5.36 GB VRAM, falling to 1.05 GB after each manual release (baseline 0.99 GB), with no growth across 3 CXRMate Multi TF ↔ CheXOne switch cycles. A release during a run returned 409 and the run completed; a 5 s idle timeout did not interrupt a run; a 20 s idle timeout released the model about 25 s after the run. In the in-app browser, the CXRMate study (2 images, image 2 selected, generated draft with an unsaved edit) survived a switch to CheXOne, a Reports round trip and Free memory without prompts. Clear study reset only the active model. Angular passed 100/100 and focused backend suites passed 141. XREPORT-checkpoint residency is unit-tested only because no checkpoint is registered locally. On 2026-11-11, manual model-notice behavior passed; 25 Inference E2E cases passed and the responsive-page test failed its model-notice assertion because its index selector reselected the default model after a custom row shifted the order. Selecting a different model manually showed the expected notice; the failure is a test assumption. |
| `ui.dataset_and_training_surfaces` | `PARTIAL` | V00–V04 browser checks on the post-split `develop` source used synthetic SQLite data: both profiles rendered, Studio imported clean/malformed/missing-text/partial CSVs, the viewer navigated image rows, dataset processing produced 7 train/1 validation rows, and pixel/text/image statistics persisted across reload. V09 completed distinct jobs and app-scoped restart checks, with cancellation and duplicate-start limits retained. V10 reviewed both apps across four viewport sizes; the Studio dataset table clipped the `Rows` column at 320px, since fixed. V05 CPU training is blocked because the pinned BEiT encoder is absent. On 2026-11-11, Studio E2E passed 27/27; the hidden in-app-browser picker prevented a complete folder-to-training journey, and the real-weight exchange test skipped. The 280×320 and 320×640 matrix passed no-overflow checks but leaves limited first-screen workspace. Packaged workflows remain open; no representative or clinical data was used. | [UI experience](ui/experience.md), [campaign](validation_campaign_ledger.md), [dataset workflow E2E](../../app/tests/e2e/test_dataset_workflow.py) |
| `test.infrastructure.suite_consolidation` | `VALIDATED` | Exact SHA `e8ad7bedd776800ee3099239d6b257ee4d5cd0c6` has 41 test source files and 419 cases: 26 Python files/331 collected cases, 14 Angular specs/79 passed, and 1 Node file/9 passed. V11 ran the live XREPORT browser suite (18/18), Studio UI/dataset/filesystem suites (14/14), 54 checkpoint/job unit tests, both client builds, and both lint targets; hosted CI 40122131254 passed. The full Python suite was collected but not executed locally. Supplemental live browser E2E on 2026-11-11 at `5e2b9932` passed Studio 27/27 and Inference 25/26; the sole failure is the model-notice assertion noted in the UI status above. The cross-app exchange case skipped because pinned model assets were absent. | [Testing and quality](coding/testing_and_quality.md), [Python suites](../../app/tests/unit/test_runtime_startup.py), [E2E suites](../../app/tests/e2e/test_angular_ui.py), [campaign](validation_campaign_ledger.md) |
| `test.infrastructure.windows_cache` | `PARTIAL` | The shared cleanup engine validates target roots and counts/removes content with a streaming .NET walk; default output is grouped and `-Verbose` includes paths. The focused PowerShell harness passes nested-tree and dry-run checks, preserve-name rules, strict locked-file handling with independent sibling cleanup, outside-root rejection, persistent-data boundaries, uninstall cache-target exclusion, and mocked MSI discovery/uninstall. A controlled 1,120-entry fixture measured 82.53 s baseline versus 1.77 s optimized for preview/counting plus deletion. Current `ClearCache`, `RemoveAllData`, and `Uninstall` dry runs passed without mutation. Junction creation was denied on this host; real cleanup, real MSI uninstall, and the full `run_tests.bat` workflow were not run. | [Testing and quality](coding/testing_and_quality.md), [startup](runtime/startup.md), [campaign](validation_campaign_ledger.md) |

## Active issues

| ID | Component | Severity | State | Current impact and next evidence |
| --- | --- | --- | --- | --- |
| `ISSUE-006` | `inference.export_safety` | `HIGH` | `PARTIAL` | 2026-11-18 audit: artificial 64x64 inputs yielded a "normal" report and an invented surgical history, exported as TXT with no research-only marking. 4.0.1: Copy and Export now start with a research-use-only notice, model and revision, source image(s) and export time (unit test plus a Playwright download assertion). Open: flag or decline unsuitable inputs before generation. |
| `ISSUE-007` | `studio.training_runtime` | `HIGH` | `OPEN` | 2026-11-18: a training job was admitted then failed at epoch 0 (`ModuleNotFoundError`, missing Keras preprocessing module; no checkpoint). Verify the official launcher repair, add a readiness check before admission, then rerun training, resume, evaluation and exchange. |
| `ISSUE-008` | `inference.study_export` | `MEDIUM` | `PARTIAL` | Combined study reports showed only the first image and exported as `xreport-<first image>.txt` without study provenance. 4.0.1: copy/export list every source image and a study exports as `xreport-study-<first image>-<n>-images.txt`. Open: the Reports card still shows only the first image. |
| `ISSUE-009` | `studio.dataset_errors` | `MEDIUM` | `OPEN` | Out-of-range validation split (`0.9`) is submitted and shown as raw 422 JSON away from the field; validate before submit with a field-level message. Also relabel the validation-results icon ("Open validation route" opens a dialog). |
| `ISSUE-010` | `studio.folder_selection` | `MEDIUM` | `OPEN` | Browser-mode folder upload opens a server-side Windows Forms dialog; in the in-app browser no chooser appeared. Verify through the foreground launcher and give browser-appropriate wording and flow. |
| `ISSUE-002` | `inference.model.medgemma` | `MEDIUM` | `BLOCKED` | The gated model cannot be installed without authorized provider access. After access is configured, install the exact pinned revision and capture real inference before changing status. |

## Evidence ownership

- Architecture and topic documents own current structures, contracts, and
  runtime behavior.
- This ledger owns component status and active issues. The
  [validation campaign ledger](validation_campaign_ledger.md) owns the concise
  gate map and release scope.
- Tests and CI runs are evidence for their stated scope, not a substitute for
  observing the browser, hardware, provider, or package boundary being claimed.
- Store temporary captures and receipts under ignored `runtimes/cache/` and
  remove them after each run. Keep source fixtures in their test fixture tree
  and the release manifest in the indexed validation documentation.
