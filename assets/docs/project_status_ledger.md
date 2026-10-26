# XREPORT Project Status Ledger

Last updated: 2026-10-26

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
The v3.2.1 candidate restores migration compatibility from published revisions
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
| `checkpoint.exchange` | `PARTIAL` | Unit coverage confirms the format-2 round trip, idempotency, conflicts, cancellation and rollback, and rejection of legacy/manifestless archives; the earlier live round trip used format 1. V08 UI on `9b298c54` rejected a lone `.keras` and a manifestless four-core ZIP before job admission. V11 adds `app/tests/e2e/test_angular_ui.py::test_checkpoint_import_rejects_manifestless_archive_in_live_ui`, which exercises the live XREPORT import dialog and verifies the typed missing-manifest error with no import progress/job. V05 Studio training is blocked by the missing pinned encoder, so positive v2 export/import, custom-model inference/restart, checksum/version/conflict UI cases, import cancellation/recovery and packaged import remain unverified. The import dialog's member list omits `manifest.json` and places metadata/history at the wrong archive paths; see `ISSUE-004`. | [Applications](architecture/applications_and_checkpoint_exchange.md), [campaign](validation_campaign_ledger.md) |
| `runtime.studio.launcher` | `PARTIAL` | Studio now has an isolated profiled environment, first-launch setup, profile reuse/switching, retryable setup state, a dedicated install action, and uninstall cleanup. Eleven focused launcher tests, PowerShell parsing, `uv lock --check`, the XREPORT CUDA-only dry-run, and both Studio profile dry-runs pass. The new first-launch installation and profile replacement were not exercised against real dependency downloads; earlier Windows launch, build freshness, process scoping, and port-release evidence remains applicable to those behaviors. | [Startup](runtime/startup.md) |
| `desktop.packaging.inference_only` | `UNVALIDATED` | PyInstaller spec and bundle audit exclude Studio modules/assets; no CPU/CUDA package was built, so package contents and import into a packaged build were not observed. | [Deployment](runtime/deployment.md) |

### Backend and persistence

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `backend.api.contracts`, `backend.jobs.lifecycle`, `backend.ml_import_boundaries` | `VALIDATED` | Covered API contracts, generic job lifecycle, and lightweight-import boundaries are validated; this does not cover every endpoint or heavy workflow. | [Backend API](architecture/backend_api.md), [API tests](../../app/tests/e2e/test_api_inference_settings.py), [job tests](../../app/tests/integration/test_job_lifecycle_api.py), [import test](../../app/tests/unit/test_runtime_startup.py) |
| `persistence.sqlite_migrations`, `persistence.inference_history`, `persistence.postgresql`, `persistence.checkpoint_registry` | `PARTIAL` | The v3.2.1 candidate passes 16 focused database-initialization tests, including upgrades from published revisions `f48a7c2e91b6` and `e91a4f6c2d73`, preserved settings/dataset rows, theme-column restoration, and repeat initialization. The migration from `b1e7a2c4d908` to `42c8e1f7b0d3` remains covered. Hosted run `39667709442` passed the PostgreSQL persistence contract, but live migration/application and authenticated switching remain unverified; live PostgreSQL testing was deferred. Crash recovery and every production configuration remain outside scope. | [Persistence](architecture/persistence.md), [SQLite tests](../../app/tests/unit/test_settings_persistence.py), [PostgreSQL contract test](../../app/tests/integration/test_persistence_contract.py) |

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
| `ui.settings`, `ui.inference.catalogue_and_sections`, `ui.reports.history`, `ui.startup_gate`, `ui.shell.routing_theme_guidance`, `ui.validation.report_review` | `VALIDATED` | Settings tabs and controls are covered by focused client/API tests and browser checks using technical fixtures: non-database changes autosave and restore after reload, while database edits remain drafts until a successful current-candidate connection test and explicit confirmation; failed probes leave the active backend unchanged. Persisted report detail edits prompt before internal navigation, including browser Back, protect reload and close where supported, block navigation during saves and deletes, and retain drafts after failed requests. Focused unit and browser coverage passed. This is not full product accessibility certification. | [UI experience](ui/experience.md), [browser E2E](../../app/tests/e2e/test_angular_ui.py) |
| `ui.inference.workflow` | `PARTIAL` | Completed inference now links its editable draft to saved history. Save patches only changed canonical report rows, including the single persisted row for study-level output. The 79-test Angular suite, focused backend coverage, history API E2E, and edit/save/reload browser journey passed locally; the browser journey used a deterministic job fixture and real SQLite history persistence. Real-provider inference and broader provider-contract coverage remain open. | [Local inference models](runtime/local_inference_models.md), [UI experience](ui/experience.md), [draft persistence specs](../../app/client/src/app/pages/inference-draft-persistence.page.spec.ts), [browser E2E](../../app/tests/e2e/test_angular_ui.py) |
| `ui.dataset_and_training_surfaces` | `PARTIAL` | V00–V04 browser checks on the post-split `develop` source used synthetic SQLite data: both profiles rendered, Studio imported clean/malformed/missing-text/partial CSVs, the viewer navigated image rows, dataset processing produced 7 train/1 validation rows, and pixel/text/image statistics persisted across reload. V09 completed distinct jobs and app-scoped restart checks, with cancellation and duplicate-start limits retained. V10 reviewed both apps across four viewport sizes; the Studio dataset table clips the `Rows` column at 320px (`ISSUE-005`). V05 CPU training is blocked because the pinned BEiT encoder is absent. Packaged workflows remain open; no representative or clinical data was used. The processing split field accepts up to 1.0 while the API caps it at 0.5 (`ISSUE-003`). | [UI experience](ui/experience.md), [campaign](validation_campaign_ledger.md), [dataset workflow E2E](../../app/tests/e2e/test_dataset_workflow.py) |
| `test.infrastructure.suite_consolidation` | `VALIDATED` | Exact SHA `e8ad7bedd776800ee3099239d6b257ee4d5cd0c6` has 41 test source files and 419 cases: 26 Python files/331 collected cases, 14 Angular specs/79 passed, and 1 Node file/9 passed. V11 ran the live XREPORT browser suite (18/18), Studio UI/dataset/filesystem suites (14/14), 54 checkpoint/job unit tests, both client builds, and both lint targets; hosted CI 40122131254 passed. The full Python suite was collected but not executed locally; focused local receipts are under `assets/QA/validation_campaign/xr-20261024-01/`. | [Testing and quality](coding/testing_and_quality.md), [Python suites](../../app/tests/unit/test_runtime_startup.py), [E2E suites](../../app/tests/e2e/test_angular_ui.py), [campaign](validation_campaign_ledger.md) |
| `test.infrastructure.windows_cache` | `PARTIAL` | Runner and direct pytest instructions use canonical `runtimes/cache` paths. Launcher cache clear and uninstall now target only current runtime/build locations; PowerShell parsing passed, but these two actions were not exercised in this run. Historical cache locations are no longer enumerated or removed. | [Testing and quality](coding/testing_and_quality.md), [startup](runtime/startup.md), [campaign](validation_campaign_ledger.md) |

## Active issues

| ID | Component | Severity | State | Current impact and next evidence |
| --- | --- | --- | --- | --- |
| `ISSUE-002` | `inference.model.medgemma` | `MEDIUM` | `BLOCKED` | The gated model cannot be installed without authorized provider access. After access is configured, install the exact pinned revision and capture real inference before changing status. |
| `ISSUE-003` | `ui.dataset.processing.validation_split` | `MEDIUM` | `OPEN` | The Studio field advertises `0–1` (`max=1`) but the backend accepts only `0.05–0.5`; entering `1.2` reaches the API and displays a typed 422. Proposed remedy: set the field maximum and Angular validation to `0.5`, then verify both boundary values. No production fix was applied during this validation campaign. |
| `ISSUE-004` | `ui.checkpoint_import.guidance` | `MEDIUM` | `OPEN` | `app/client/src/app/components/checkpoint-import.component.ts` describes a ZIP containing only four core files, omitting required `manifest.json`, and lists `metadata.json`/`session_history.json` at incorrect root paths. V11 live UI regression confirms the correctly rooted manifestless archive fails before import progress begins. Proposed remedy: describe the exported `.xreportckpt` contract and its manifest/member layout; no production copy change was applied. |
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
