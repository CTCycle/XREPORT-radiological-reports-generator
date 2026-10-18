# XREPORT Project Status Ledger

Last updated: 2026-10-18

This is the canonical current-status summary for XREPORT. It records scoped
component confidence, active issues, and known limits; it is not a development
diary, detailed test report, release approval, or clinical-quality statement.

The published v3.2.0 release was validated at source
`a8d75a0c0d5f02a29c85e9ac58f4837fc3081449`. Before this documentation update,
the `develop` branch HEAD was `1fc9abdd7a857081a44405cc2f52f698779d5cb9`, with
later repository and validation-tooling changes than the release source. The
release evidence does not imply a fresh campaign run on a newer candidate;
re-run the applicable gates before claiming that revision has passed. The
[validation campaign ledger](validation_campaign_ledger.md) holds release
evidence and campaign-wide boundaries.

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
| `runtime.source.startup` | `PARTIAL` | Warm launch, readiness, isolated database startup, and selected recovery paths are exercised. Interactive process-ownership races and rebuild invalidation remain open. | [Startup](runtime/startup.md) |
| `runtime.desktop.packaged` | `PARTIAL` | Published v3.2.0 CPU/CUDA portable and MSI packages passed the scoped acceptance described in the campaign ledger. CUDA fallback without an NVIDIA GPU and cross-variant performance remain untested. | [Campaign](validation_campaign_ledger.md), [deployment](runtime/deployment.md) |
| `runtime.containerized` | `NOT_IMPLEMENTED` | No supported container image or deployment contract exists. | [Runtime modes](runtime/modes.md) |
| `configuration.runtime_settings` | `VALIDATED` | Public database-backed settings, limits, save/reload/reset, and hidden-field protection are covered. Infrastructure and static model-policy values are not user settings. | [Configuration](runtime/configuration.md) |
| `security.authentication` | `NOT_IMPLEMENTED` | The current boundary is trusted local use; external exposure is not covered. | [Execution and data flow](architecture/execution_and_data_flow.md) |

### Backend and persistence

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `backend.api.contracts`, `backend.jobs.lifecycle`, `backend.ml_import_boundaries` | `VALIDATED` | Covered API contracts, generic job lifecycle, and lightweight-import boundaries are validated; this does not cover every endpoint or heavy workflow. | [Backend API](architecture/backend_api.md), [API tests](../../app/tests/e2e/test_inference_api.py), [job tests](../../app/tests/integration/test_job_lifecycle_api.py), [import test](../../app/tests/unit/test_ml_import_boundaries.py) |
| `persistence.sqlite_migrations`, `persistence.inference_history`, `persistence.postgresql`, `persistence.checkpoint_registry` | `VALIDATED` | SQLite migrations/restart, PostgreSQL 16 contract, inference history, and checkpoint registry behavior are covered for the exercised schemas and fixtures. Crash recovery and every production configuration are outside this scope. | [Persistence](architecture/persistence.md), [SQLite tests](../../app/tests/unit/test_database_initialization.py), [PostgreSQL contract test](../../app/tests/integration/test_persistence_contract.py) |

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
| `inference.model.cxrmate-multi` | `PARTIAL` | Pinned technical inference and packaged CPU/CUDA inference are covered; model `validation_status` remains `pending`, with no quality or clinical claim. | [Local inference models](runtime/local_inference_models.md), [campaign](validation_campaign_ledger.md) |
| `inference.model.cxrmate-ed` | `DEPRECATED` | Retired after three distinct fixtures collapsed to the same full report. Historical user data and model files remain preserved. | [Local inference models](runtime/local_inference_models.md) |
| `inference.model.chexone`, `inference.model.cxrmate-2` | `VALIDATED` | Pinned technical generation and conservative output review passed for bounded fixtures; diagnostic accuracy is not established. | [Local inference models](runtime/local_inference_models.md) |
| `inference.model.medgemma` | `BLOCKED` | Gated-provider behavior is validated, but real installation and inference require authorized access. | [Gated access](runtime/local_inference_models.md); `ISSUE-002` below |

### User interface and test infrastructure

| Component | Status | Current scope or limit | Authority |
| --- | --- | --- | --- |
| `ui.settings`, `ui.inference.catalogue_and_sections`, `ui.reports.history`, `ui.startup_gate`, `ui.shell.routing_theme_guidance`, `ui.validation.report_review` | `VALIDATED` | The named routes, states, persistence, and report review are covered by focused client/API tests and browser checks using technical fixtures. This is not full product accessibility certification. | [UI experience](ui/experience.md), [browser E2E](../../app/tests/e2e/test_angular_ui.py) |
| `ui.inference.workflow` | `PARTIAL` | Generate/review is observed for one provider and limited inputs. Other provider contracts and browser download-event observation remain open. | [Local inference models](runtime/local_inference_models.md), [browser E2E](../../app/tests/e2e/test_angular_ui.py) |
| `ui.dataset_and_training_surfaces` | `PARTIAL` | The routes and selected technical flows are exercised; representative data, full packaged workflows, and spoken assistive-technology output remain outside. | [UI experience](ui/experience.md), [dataset workflow E2E](../../app/tests/e2e/test_dataset_workflow_ui.py) |
| `test.infrastructure.windows_cache` | `PARTIAL` | The official runner uses disposable configured caches. Some pre-existing protected cache paths still produce access warnings and were left untouched. | [Testing and quality](coding/testing_and_quality.md), [startup](runtime/startup.md) |

## Active issues

| ID | Component | Severity | State | Current impact and next evidence |
| --- | --- | --- | --- | --- |
| `ISSUE-002` | `inference.model.medgemma` | `MEDIUM` | `BLOCKED` | The gated model cannot be installed without authorized provider access. After access is configured, install the exact pinned revision and capture real inference before changing status. |
| `ISSUE-004` | `test.infrastructure.windows_cache` | `LOW` | `OPEN` | Git scans can encounter protected legacy cache paths. Keep using the configured runner; inspect ownership and access before any cleanup. |

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
