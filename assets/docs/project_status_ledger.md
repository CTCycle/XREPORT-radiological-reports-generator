# XREPORT Project Status Ledger

Last updated: 2026-10-21

This is the canonical current-status summary for XREPORT. It records scoped
component confidence, active issues, and known limits; it is not a development
diary, detailed test report, release approval, or clinical-quality statement.

The published v3.2.0 release was validated at source
`a8d75a0c0d5f02a29c85e9ac58f4837fc3081449`. The most recent application-source
Tier 0 validation targeted `b98a9b60492b63fb41a904e3cff4815325b8d075`. On
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
seed `317` survived navigation and launcher restart; backend health was OK,
SQLite integrity passed, and Alembic reached `42c8e1f7b0d3`. The v3.2.1
candidate restores migration compatibility from published revisions
`f48a7c2e91b6` and `e91a4f6c2d73`, including the prior browser theme
preference. Sixteen focused database-initialization tests, four version/OpenAPI
tests, 65 client unit tests, and both lint gates passed. The smoke preceded the
3.2.1 metadata bump. Hosted run `39662773417` passed on the preceding develop
revision; candidate CI and official packages remain open. Tier 0 remains
`PARTIAL` until those gates finish. Follow-on cleanup removed the neutral validation
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
| `runtime.source.startup` | `PARTIAL` | The isolated launcher manifest evaluator passed current, stale-source, stale-dependency, missing, invalid, and missing-output cases; occupied-port fail-closed and decline paths preserved the unrelated listener. Current-source startup rendered Inference, Reports, Dataset, Training, and Settings; seed `317` survived navigation and restart with isolated resources. Backend health, SQLite integrity, and Alembic head `42c8e1f7b0d3` passed. Focused migration, version/OpenAPI, client-unit, and lint checks passed; candidate hosted CI remains pending. Earlier S02 state-transition coverage and the disposable-port launcher smoke remain valid for their stated scopes. | [Startup](runtime/startup.md), [campaign](validation_campaign_ledger.md) |
| `runtime.desktop.packaged` | `PARTIAL` | Published v3.2.0 CPU/CUDA portable and MSI packages passed the scoped acceptance described in the campaign ledger. CUDA fallback without an NVIDIA GPU and cross-variant performance remain untested. | [Campaign](validation_campaign_ledger.md), [deployment](runtime/deployment.md) |
| `runtime.containerized` | `NOT_IMPLEMENTED` | No supported container image or deployment contract exists. | [Runtime modes](runtime/modes.md) |
| `configuration.runtime_settings` | `VALIDATED` | On `ef1ae34ad8e9dc073415e5ec1a0eaec588f81999`, rendered settings tabs passed autosave, bounds, reset, reload/restart persistence, per-database isolation, failed-connection draft handling, and the Settings API policy boundary. Supported `.env` parsing and CORS behavior passed isolated fixtures. The ignored local `settings/.env` was not changed during the settings checks. | [Configuration](runtime/configuration.md), [campaign](validation_campaign_ledger.md), [environment tests](../../app/tests/unit/test_runtime_environment.py), [settings E2E](../../app/tests/e2e/test_angular_ui.py) |
| `configuration.database_runtime` | `PARTIAL` | Database values remain uncommitted drafts until the current candidate passes a connection test and receives explicit confirmation; SQLite settings isolation, restart persistence, failed PostgreSQL handling, and rollback passed. The request-drain gate test did not reach its test body because Windows asyncio loop initialization stalled. A successful authenticated PostgreSQL connection and live switch remain unverified; the user explicitly deferred PostgreSQL testing on 2026-10-21, so no connection or switch was attempted in this run. | [Configuration](runtime/configuration.md), [Database configuration tests](../../app/tests/unit/test_database_configuration.py) |
| `security.authentication` | `NOT_IMPLEMENTED` | The current boundary is trusted local use; external exposure is not covered. | [Execution and data flow](architecture/execution_and_data_flow.md) |

### Backend and persistence

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `backend.api.contracts`, `backend.jobs.lifecycle`, `backend.ml_import_boundaries` | `VALIDATED` | Covered API contracts, generic job lifecycle, and lightweight-import boundaries are validated; this does not cover every endpoint or heavy workflow. | [Backend API](architecture/backend_api.md), [API tests](../../app/tests/e2e/test_inference_api.py), [job tests](../../app/tests/integration/test_job_lifecycle_api.py), [import test](../../app/tests/unit/test_ml_import_boundaries.py) |
| `persistence.sqlite_migrations`, `persistence.inference_history`, `persistence.postgresql`, `persistence.checkpoint_registry` | `PARTIAL` | The v3.2.1 candidate passes 16 focused database-initialization tests, including upgrades from published revisions `f48a7c2e91b6` and `e91a4f6c2d73`, preserved settings/dataset rows, theme-column restoration, and repeat initialization. The migration from `b1e7a2c4d908` to `42c8e1f7b0d3` remains covered. PostgreSQL check-normalization unit coverage passes, but live migration/application and authenticated switching remain unverified; live PostgreSQL testing was deferred. Candidate CI is pending. Crash recovery and every production configuration remain outside scope. | [Persistence](architecture/persistence.md), [SQLite tests](../../app/tests/unit/test_database_initialization.py), [PostgreSQL contract test](../../app/tests/integration/test_persistence_contract.py) |

### User workflows

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `workflow.dataset_upload_and_preparation` | `VALIDATED` | Synthetic upload, matching, confirmation, processing, and deletion protections are exercised. Representative-data, large-scale, and packaged workflows are not established. | [Workflows](operations/workflows.md), [dataset E2E](../../app/tests/e2e/test_dataset_workflow_api.py) |
| `workflow.training_and_resume` | `VALIDATED` | Minimal local training, cancellation, resume, and checkpoint persistence are exercised with technical fixtures. Model quality and representative training performance are not established. | [Workflows](operations/workflows.md), [training E2E](../../app/tests/e2e/test_training_api.py) |
| `workflow.dataset_validation`, `workflow.checkpoint_evaluation` | `VALIDATED` | Synthetic technical validation and checkpoint evaluation complete and persist reports. No clinical or representative-data quality claim follows. | [Workflows](operations/workflows.md), [validation tests](../../app/tests/unit/test_validation_contract_metrics.py) |
| `workflow.inference.generation_pipeline` | `VALIDATED` | The end-to-end contract is exercised with pinned local models and synthetic/de-identified inputs; provider-wide quality and all failure/reuse conditions remain outside. | [Local inference models](runtime/local_inference_models.md), [inference E2E](../../app/tests/e2e/test_inference_api.py) |

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
| `ui.settings`, `ui.inference.catalogue_and_sections`, `ui.reports.history`, `ui.startup_gate`, `ui.shell.routing_theme_guidance`, `ui.validation.report_review` | `VALIDATED` | Settings tabs and controls are covered by focused client/API tests and browser checks using technical fixtures: non-database changes autosave and restore after reload, while database edits remain drafts until a successful current-candidate connection test and explicit confirmation; failed probes leave the active backend unchanged. This is not full product accessibility certification. | [UI experience](ui/experience.md), [browser E2E](../../app/tests/e2e/test_angular_ui.py) |
| `ui.inference.workflow` | `PARTIAL` | Generate/review is observed for one provider and limited inputs. Other provider contracts and browser download-event observation remain open. | [Local inference models](runtime/local_inference_models.md), [browser E2E](../../app/tests/e2e/test_angular_ui.py) |
| `ui.dataset_and_training_surfaces` | `PARTIAL` | The routes and selected technical flows are exercised; representative data, full packaged workflows, and spoken assistive-technology output remain outside. | [UI experience](ui/experience.md), [dataset workflow E2E](../../app/tests/e2e/test_dataset_workflow_ui.py) |
| `test.infrastructure.windows_cache` | `PARTIAL` | Runner and direct pytest instructions use canonical `runtimes/cache` paths. Launcher cache clear and uninstall now target only current runtime/build locations; PowerShell parsing passed, but these two actions were not exercised in this run. Historical cache locations are no longer enumerated or removed. | [Testing and quality](coding/testing_and_quality.md), [startup](runtime/startup.md), [campaign](validation_campaign_ledger.md) |

## Active issues

| ID | Component | Severity | State | Current impact and next evidence |
| --- | --- | --- | --- | --- |
| `ISSUE-002` | `inference.model.medgemma` | `MEDIUM` | `BLOCKED` | The gated model cannot be installed without authorized provider access. After access is configured, install the exact pinned revision and capture real inference before changing status. |

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
