# XREPORT Project Status Ledger

Last updated: 2026-09-21

This is the canonical high-level operational status catalog for the current
XREPORT checkout. It summarizes what is working, validated, partial, blocked,
unvalidated, or absent, and points to the detailed architecture and QA evidence
that supports each claim. It is a current-state index, not a development diary,
issue tracker replacement, release approval, or clinical-quality statement.

Snapshot: `develop` at `6c38954` (`qa: record real CheXOne validation receipt`).
Statuses describe the evidence available for that checkout and must be refreshed
when the checkout or its validation evidence changes.

## Maintenance Rules

Future coding and validation agents must:

1. inspect this ledger before substantial implementation or validation work;
2. use it to find known defects and previously validated behavior;
3. update affected entries after implementation;
4. update evidence after meaningful tests or manual validation;
5. never mark a component `VALIDATED` without evidence appropriate to its scope;
6. downgrade a status when a regression is discovered;
7. close or archive an issue only after remediation and successful revalidation;
8. avoid duplicate issue entries for the same underlying defect;
9. link detailed reports instead of copying long narratives into this ledger;
10. keep this ledger synchronized with the actual repository state.

Evidence under `assets/QA/` is often local and ignored by Git. A link is valid
for this checkout, but if the artifact is absent in another checkout, its claim
must be treated as unvalidated until the evidence is restored or the check is
rerun. The ledger never upgrades a claim merely because source code or a unit
test exists.

## Status Taxonomy

| Status | Meaning |
| --- | --- |
| `VALIDATED` | Implemented and confirmed through meaningful testing or observation appropriate to the stated scope. |
| `WORKING` | Believed to work from implementation and limited or technical-path testing, but not fully validated. |
| `PARTIAL` | Implemented but incomplete, degraded, or confirmed only for part of the expected behavior. |
| `BROKEN` | Known not to work correctly for the stated scope. |
| `BLOCKED` | Cannot currently be validated or completed because of an external dependency, credential, service, hardware, or similar blocker. |
| `UNVALIDATED` | Implementation exists, but available evidence is insufficient to claim that it works. |
| `NOT_IMPLEMENTED` | The expected capability is currently absent. |
| `DEPRECATED` | Intentionally retained only for compatibility or scheduled for removal. |

`VALIDATED` is scoped: passing unit tests may validate a helper or contract,
but does not validate an end-to-end workflow. `Severity` in the issue catalog is
independent from component status. Validation levels used below are `None`,
`unit`, `integration`, `E2E`, and `manual`; combined values mean that each named
form of evidence exists for the stated scope.

## Current Snapshot

- Source-mode backend startup, lightweight API reachability, and the rendered
  Angular smoke surface have recent evidence.
- Settings persistence and reset have browser and automated evidence from
  2026-09-17; responsive settings, dataset, and inference evidence was captured
  again on 2026-09-19/20.
- Real single-fixture technical inference has been observed for the three
  CXRMate public models on 2026-09-19 and CheXOne on 2026-09-20. These receipts
  do not establish clinical quality or catalog validation promotion.
- CXRMate-ED has an active degraded three-case sensitivity finding. MedGemma is
  access-blocked by its gated provider terms and credential requirement.
- Full dataset, training, successful validation/evaluation, PostgreSQL, and
  packaged desktop workflows remain validation debt even where implementation
  and focused tests exist.
- No current component is classified `BROKEN` from the inspected evidence.
  Historical failures that were superseded are listed separately below.

## Current Component Ledger

### Runtime, configuration, and platform

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `runtime.source.startup` | `VALIDATED` | Source-mode FastAPI startup, SQLite migration/resource checks, health readiness, and lightweight endpoint reachability. | [2026-09-19 validation note](../QA/xreport-validation-2026-09-19.md); [backend smoke log](../QA/xreport-backend-smoke-20260919.log) | Checkpoint listing emits warnings for incomplete test artifacts; see `ISSUE-003`. | — | 2026-09-19 | integration + E2E + manual | [startup](runtime/startup.md); [system overview](architecture/system_overview.md) | Repeat the source smoke after launcher, migration, or startup changes. |
| `runtime.desktop.packaged` | `UNVALIDATED` | Tauri CPU/CUDA runtime extraction, portable/MSI packaging, startup, shutdown, and user-data-root isolation. | [desktop packaging tests](../../app/tests/unit/test_desktop_packaging.py); [2026-09-17 E2E note](../QA/e2e-validation-20260917.md) records `cargo check` passed. | No current CPU/CUDA portable/MSI smoke report is present under `assets/QA/desktop/`. | — | 2026-09-17 | unit + build-check | [deployment](runtime/deployment.md); [runtime modes](runtime/modes.md) | Build both variants and run the documented packaged smoke checks, preserving reports under `assets/QA/desktop/`. |
| `runtime.containerized` | `NOT_IMPLEMENTED` | Containerized runtime or image-based deployment. | [runtime modes](runtime/modes.md) explicitly records this mode as absent. | No container build, image, or deployment contract exists. | — | — | None | [runtime modes](runtime/modes.md); [deployment](runtime/deployment.md) | Define a supported container contract only if deployment scope expands. |
| `configuration.runtime_settings` | `VALIDATED` | Database-backed public settings: seed, filesystem access, polling interval, and inference timeout, including save, reload, reset, and allowlisting. | [settings validation](../QA/settings-migration-validation-20260917.md); [settings E2E](../../app/tests/e2e/test_settings_api.py); [settings page E2E](../../app/tests/e2e/test_angular_ui.py) | Hidden infrastructure, credential, and static model-policy values remain intentionally unavailable to the Settings API. | — | 2026-09-17 | unit + integration + E2E + manual | [configuration](runtime/configuration.md); [persistence](architecture/persistence.md); [UI experience](ui/experience.md) | Revalidate save/reset and migration behavior after settings or schema changes. |
| `security.authentication` | `NOT_IMPLEMENTED` | API authentication and authorization. | [execution and data flow](architecture/execution_and_data_flow.md) states that no auth layer is implemented. | Current security boundary is trusted local use; external exposure is not covered. | — | — | None | [execution and data flow](architecture/execution_and_data_flow.md); [runtime modes](runtime/modes.md) | Define and validate a threat model before supporting non-local deployment. |

### Backend, jobs, and persistence

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `backend.api.contracts` | `VALIDATED` | Tested health, model catalogue, dataset status/name, checkpoint, and settings surfaces plus typed error behavior. | [2026-09-19 validation note](../QA/xreport-validation-2026-09-19.md); [backend API tests](../../app/tests/e2e/test_inference_api.py); [OpenAPI tests](../../app/tests/unit/test_openapi_schema.py) | This status covers the exercised surface, not every endpoint or every long-running happy path. | — | 2026-09-19 | unit + integration + E2E | [backend API](architecture/backend_api.md); [system overview](architecture/system_overview.md) | Extend endpoint-level E2E coverage when a route or response contract changes. |
| `backend.jobs.lifecycle` | `VALIDATED` | Generic start, poll, cancellation, terminal failure, recoverability, and persistence-failure semantics. | [architecture review](architecture/architecture_review.md); [job failure tests](../../app/tests/unit/test_job_failure_semantics.py); [job cancellation tests](../../app/tests/unit/test_job_cancellation_semantics.py) | No current issue recorded for the generic lifecycle contract. | — | 2026-09-17 | unit | [execution and data flow](architecture/execution_and_data_flow.md); [backend API](architecture/backend_api.md) | Revalidate the affected job path after changes to a feature service or polling contract. |
| `backend.ml_import_boundaries` | `VALIDATED` | Lightweight endpoints avoid eager Keras/PyTorch/Transformers/provider imports and remain usable before an ML job. | [clean backend smoke](../QA/xreport-backend-smoke-20260919.log) has no deadlock/partial-import/HTTP-500 matches; [import-boundary tests](../../app/tests/unit/test_ml_import_boundaries.py) | The rule is a regression boundary; actual model execution remains provider-specific below. | — | 2026-09-19 | unit + integration + manual | [execution and data flow](architecture/execution_and_data_flow.md); [troubleshooting](operations/troubleshooting.md) | Repeat the clean-subprocess and lightweight-endpoint checks after model/service import changes. |
| `persistence.sqlite_migrations` | `VALIDATED` | SQLite startup/initialization reaches the checked-in Alembic head and persists application settings without implicit schema stamping. | [backend smoke](../QA/xreport-backend-smoke-20260919.log) records head `f48a7c2e91b6`; [database initialization tests](../../app/tests/unit/test_database_initialization.py); [settings migration validation](../QA/settings-migration-validation-20260917.md) | Existing unversioned or incompatible schemas intentionally fail closed. | — | 2026-09-19 | unit + integration + manual | [persistence](architecture/persistence.md); [architecture review](architecture/architecture_review.md) | Recheck migration upgrade and rollback safety for every new revision. |
| `persistence.postgresql` | `UNVALIDATED` | External PostgreSQL creation, locking, migration, and repository contract. | [PostgreSQL contract test](../../app/tests/integration/test_persistence_contract.py) requires external test settings; no current PostgreSQL QA artifact was found. | No current live PostgreSQL evidence. | — | — | None | [persistence](architecture/persistence.md); [deployment](runtime/deployment.md) | Run the integration contract against the supported PostgreSQL versions/configuration. |
| `persistence.checkpoint_registry` | `PARTIAL` | Database-owned checkpoint identity, complete-artifact registration, listing, and safe deletion. | [backend smoke](../QA/xreport-backend-smoke-20260919.log); [checkpoint/deletion tests](../../app/tests/e2e/test_training_api.py) | Four incomplete `e2e_delete_*` registrations produce warnings while the listing endpoint still returns HTTP 200; see `ISSUE-003`. | — | 2026-09-19 | unit + integration + manual | [persistence](architecture/persistence.md); [backend API](architecture/backend_api.md) | Reconcile incomplete fixture registrations and rerun startup plus checkpoint listing. |

### Domain workflows

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `workflow.dataset_upload_and_preparation` | `UNVALIDATED` | Non-empty upload, image matching, unmatched-row confirmation, processing, persistence, and downstream-ready dataset. | [upload E2E tests](../../app/tests/e2e/test_upload_api.py); [image scanning tests](../../app/tests/unit/test_preparation_image_scanning.py) | Current evidence covers parsers, guards, and route surfaces, not a complete live happy path. | — | 2026-09-17 | unit + endpoint E2E | [workflows](operations/workflows.md); [backend API](architecture/backend_api.md) | Run a disposable non-empty dataset through load, partial-confirmation, processing, and persisted metadata checks. |
| `workflow.training_and_resume` | `UNVALIDATED` | Real training, checkpoint creation/registration, stop, resume, and later model use. | [training API tests](../../app/tests/e2e/test_training_api.py); [training worker tests](../../app/tests/unit/test_training_stop_mechanism.py); [training memory tests](../../app/tests/unit/test_training_memory_guards.py) | Route and guard evidence exists, but no recent real training/resume receipt is recorded. | — | 2026-09-17 | unit + endpoint E2E | [workflows](operations/workflows.md); [execution and data flow](architecture/execution_and_data_flow.md) | Run the documented minimal CPU training smoke, then verify checkpoint metadata, resume, cancellation, and cleanup. |
| `workflow.dataset_validation` | `PARTIAL` | Successful dataset validation, metric artifact persistence, and report review. | [validation route E2E](../../app/tests/e2e/test_angular_ui.py) confirms the missing-dataset error path; [validation contract tests](../../app/tests/unit/test_validation_contract_metrics.py) cover metric bounds. | A successful non-empty validation run and report review are not in current QA evidence. | — | 2026-09-17 | unit + E2E error path | [workflows](operations/workflows.md); [backend API](architecture/backend_api.md) | Validate a prepared dataset through completion and inspect the saved report/artifacts. |
| `workflow.checkpoint_evaluation` | `UNVALIDATED` | Compatible checkpoint evaluation, associated dataset resolution, metrics, and report persistence. | [evaluation tests](../../app/tests/unit/test_evaluation.py); [validation job tests](../../app/tests/unit/test_validation_job_semantics.py) | Preflight and failure semantics are tested; a successful live evaluation is not evidenced. | — | 2026-09-17 | unit | [workflows](operations/workflows.md); [backend API](architecture/backend_api.md) | Run a compatible checkpoint evaluation and retrieve its persisted report. |
| `workflow.inference.generation_pipeline` | `WORKING` | Local model installation/readiness, study generation, job completion, provenance, and report persistence. | [public-model run summary](../QA/inference_validation_runs/public-inference-models-20260919T165746Z.json); [CheXOne receipt](../QA/inference_validation/StanfordAIMI__CheXOne-0c350e6852ea08f9d9baf3b7595c1a10d4849927.json); [provider tests](../../app/tests/unit/test_huggingface_provider.py) | Single-fixture technical receipts do not validate quality, all user workflows, or every provider. | — | 2026-09-20 | unit + integration + manual | [local inference models](runtime/local_inference_models.md); [workflows](operations/workflows.md) | Revalidate installation reuse, cancel/timeout, persistence, and multi-case quality for each supported model. |

### Model providers

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `inference.catalogue` | `WORKING` | Five pinned public entries, exact revisions, adapters, access/readiness state, and model-declared output sections. | [catalogue tests](../../app/tests/unit/test_inference_model_catalog.py); [current inference API test](../../app/tests/e2e/test_inference_api.py); [model configuration](../../app/server/configurations/inference_models.py) | The five-model aggregate run on 2026-09-19 predates the CheXOne findings-only contract fix; rerun against this checkout is still needed. | — | 2026-09-20 | integration + manual | [local inference models](runtime/local_inference_models.md); [backend API](architecture/backend_api.md) | Rerun the aggregate catalogue contract check after the current CheXOne change. |
| `inference.model.cxrmate-multi` | `WORKING` | Pinned CXRMate Multi TF single-fixture generation with Findings and Impression output. | [2026-09-19 receipt](../QA/inference_validation/aehrc__cxrmate-multi-tf-330721b9aa5bba201a3eb88eba4dd9a6607f3e7a.json) records a passed real inference. | Catalogue `validation_status` remains `pending`; technical generation is not quality validation. | — | 2026-09-19 | integration + manual | [local inference models](runtime/local_inference_models.md) | Run a multi-case quality gate and verify readiness/provenance promotion rules. |
| `inference.model.cxrmate-ed` | `PARTIAL` | Pinned CXRMate-ED generation with clinical context and Findings/Impression output. | [2026-09-19 receipt](../QA/inference_validation/aehrc__cxrmate-ed-68251c7605067ddbea330413aade032713fd2192.json); [failed three-case canary](../QA/inference_validation_runs/cxrmate-ed-sensitivity-20260919T154916Z.json) | Three real cases completed but only two report texts were distinct; catalogue status is `degraded`; see `ISSUE-001`. | — | 2026-09-19 | integration + manual | [local inference models](runtime/local_inference_models.md); [commands](operations/commands_and_locations.md) | Remediate or accept the canary finding, then rerun the full three-case gate. |
| `inference.model.chexone` | `WORKING` | Pinned CheXOne Findings-only report generation and declared UI contract. | [2026-09-20 receipt](../QA/inference_validation/StanfordAIMI__CheXOne-0c350e6852ea08f9d9baf3b7595c1a10d4849927.json); [CheXOne contract tests](../../app/tests/e2e/test_inference_api.py) | Receipt status is passed for one fixture, but catalogue `validation_status` is `pending` and `manifest_promoted` is false. | — | 2026-09-20 | integration + manual | [local inference models](runtime/local_inference_models.md); [UI patterns](ui/components_and_patterns.md) | Rerun the current five-model aggregate and a multi-case quality gate. |
| `inference.model.cxrmate-2` | `WORKING` | Pinned CXRMate-2 Findings/Impression generation on a high-demand local runtime. | [2026-09-19 receipt](../QA/inference_validation/aehrc__cxrmate-2-aa8e2d16470e20671acf049687b4707c9bf2f2b5.json) records a passed real inference. | Catalogue `validation_status` remains `pending`; high memory/time demand is documented, not a confirmed defect. | — | 2026-09-19 | integration + manual | [local inference models](runtime/local_inference_models.md); [runtime configuration](runtime/configuration.md) | Run a multi-case quality and resource-budget check before promoting readiness. |
| `inference.model.medgemma` | `BLOCKED` | Gated MedGemma installation and local generation. | [public-model run summary](../QA/inference_validation_runs/public-inference-models-20260919T165746Z.json) records `deferred_access_required`. | Fifth public model is visible but cannot be downloaded or generated in the current environment; see `ISSUE-002`. | Provider terms acceptance plus `HF_TOKEN` or an approved local Hugging Face credential. | 2026-09-19 | manual access check | [gated access](runtime/local_inference_models.md); [configuration](runtime/configuration.md) | After access is granted, download the exact pinned revision and capture a real inference receipt. |

### User interface and test infrastructure

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ui.settings` | `VALIDATED` | Settings route navigation, four public controls, save/reload/reset, explicit states, and responsive layout. | [settings E2E](../../app/tests/e2e/test_angular_ui.py); [2026-09-17 E2E note](../QA/e2e-validation-20260917.md); [responsive screenshots](../QA/xreport-settings-narrow-dark.png) | No current issue recorded for the exercised settings surface. | — | 2026-09-20 | E2E + manual | [UI experience](ui/experience.md); [UI patterns](ui/components_and_patterns.md) | Revalidate persistence and responsive states after route, contract, or token changes. |
| `ui.inference.catalogue_and_sections` | `VALIDATED` | Rendered inference route, model-card grid, banner behavior, responsive layout, and model-declared report sections. | [2026-09-19 validation note](../QA/xreport-validation-2026-09-19.md); [inference screenshot](../QA/xreport-inference-narrow-dark.png); [CheXOne UI contract test](../../app/tests/e2e/test_angular_ui.py) | Live browser Generate/Cancel/edit/export coverage is not established for every provider. | — | 2026-09-20 | E2E + manual | [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Add a rendered Generate/Cancel/provenance flow against a current installed model. |
| `ui.startup_gate` | `VALIDATED` | Shell-level startup surface, serialized `/api/health` readiness polling, route suppression before readiness, slow/unavailable recovery, ready transition, and responsive 1024x720 composition. | [startup screenshot](../QA/xreport-startup-gate.png); [startup E2E](../../app/tests/e2e/test_angular_ui.py); [frontend unit suite](../../app/client/src/app/services/startup-readiness.service.spec.ts) | Real `start_on_windows.ps1 -Action Launch` timing and packaged desktop smoke remain separate validation work. | — | 2026-09-21 | unit + E2E + manual | [startup](runtime/startup.md); [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Recheck launcher timing and packaged CPU/CUDA startup after the next desktop validation run. |
| `ui.dataset_and_training_surfaces` | `WORKING` | Dataset and Training route rendering, navigation, and responsive surface. | [route-render E2E](../../app/tests/e2e/test_angular_ui.py); [dataset screenshot](../QA/xreport-dataset-narrow-dark.png) | Route rendering is evidenced, but complete dataset and training workflows remain unvalidated. | — | 2026-09-20 | E2E + manual | [UI patterns](ui/components_and_patterns.md); [workflows](operations/workflows.md) | Pair the surface checks with live dataset and minimal-training workflow evidence. |
| `ui.validation.report_review` | `PARTIAL` | Validation route, missing-data error feedback, successful report display, and metric review. | [missing-dataset E2E](../../app/tests/e2e/test_angular_ui.py); [validation component patterns](ui/components_and_patterns.md) | Error-path rendering is covered; successful report review is not evidenced. | — | 2026-09-17 | E2E error path | [UI experience](ui/experience.md); [workflows](operations/workflows.md) | Run a successful validation and inspect the rendered report/metric state. |
| `test.infrastructure.windows_cache` | `PARTIAL` | Repeatable Windows test startup and disposable-cache routing. | [testing rules](coding/testing_and_quality.md); [settings validation](../QA/settings-migration-validation-20260917.md); [E2E validation](../QA/e2e-validation-20260917.md) | Passing runs still emit warnings for pre-existing protected pytest/cache directories; see `ISSUE-004`. | Existing filesystem ACLs on protected cache paths. | 2026-09-19 | integration + E2E | [startup](runtime/startup.md); [commands](operations/commands_and_locations.md) | Rerun the test runner after routing/cleanup and confirm warnings are removed or explicitly accepted. |

## Open Issues

Severity is deliberately separate from functional status: a `PARTIAL` component
may have a `LOW` issue, and a `BLOCKED` component may have no software defect.
Only actionable current problems belong here.

| ID | Affected Component | Severity | Concise Description | Current Impact | Reproduction or Evidence | Suspected Cause | Blocker | Remediation Status | Required Revalidation | Related Documentation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ISSUE-001` | `inference.model.cxrmate-ed` | `HIGH` | The real three-case sensitivity canary completed but produced only two distinct report texts, so the catalogue remains `degraded`. | CXRMate-ED remains selectable only as an unverified research draft; quality promotion is not justified. | [canary JSON](../QA/inference_validation_runs/cxrmate-ed-sensitivity-20260919T154916Z.json): `completed_cases=3`, `reports_all_distinct=false`, `unique_report_count=2`. | Not established by the evidence. | — | `OPEN` — preserve the degraded warning and do not promote the manifest. | Rerun the pinned three-case canary after remediation and review all outputs against the quality gate. | [local inference models](runtime/local_inference_models.md); [canary command](operations/commands_and_locations.md) |
| `ISSUE-002` | `inference.model.medgemma` | `MEDIUM` | The gated MedGemma entry cannot be installed or generated without provider terms acceptance and a Hugging Face credential. | One of the five catalogue entries is unavailable in the current environment. | [public-model run](../QA/inference_validation_runs/public-inference-models-20260919T165746Z.json): `state=deferred_access_required`. | Not a confirmed software defect. | Provider terms plus `HF_TOKEN` or an approved local credential. | `BLOCKED` — waiting for authorized access configuration. | Download the exact pinned revision, run real inference, and capture a receipt before changing status. | [gated access](runtime/local_inference_models.md); [configuration](runtime/configuration.md) |
| `ISSUE-003` | `persistence.checkpoint_registry` | `LOW` | Clean backend smoke lists four incomplete `e2e_delete_*` checkpoint registrations and logs warnings. | The checkpoint endpoint still returns HTTP 200, but the catalogue is noisy and may expose stale test registrations. | [backend smoke log](../QA/xreport-backend-smoke-20260919.log) shows the four warning names during checkpoint listing. | Stale test-generated registrations are plausible but not confirmed. | — | `OPEN` — reconcile incomplete registrations before treating checkpoint listing as clean. | Remove or repair the fixtures in a disposable database, rerun startup/listing, and confirm zero warnings. | [persistence](architecture/persistence.md); [backend API](architecture/backend_api.md) |
| `ISSUE-004` | `test.infrastructure.windows_cache` | `LOW` | Test/E2E runs pass but emit permission warnings for pre-existing protected pytest/cache directories. | Results remain usable, but cleanup and reproducibility are less clear on Windows. | [settings QA](../QA/settings-migration-validation-20260917.md) and [E2E QA](../QA/e2e-validation-20260917.md) record the warnings. | Protected pre-existing cache ACLs, as documented by the validation runs. | Filesystem permission/ownership of the legacy cache paths. | `OPEN` — keep active caches under `runtimes/cache` and report any protected legacy paths explicitly. | Rerun the relevant tests and confirm no unexpected cache-permission warnings remain. | [testing rules](coding/testing_and_quality.md); [startup](runtime/startup.md) |

## Validation Debt

Validation debt is not a defect list. It identifies important behavior whose
current confidence is too low or too narrow to support a stronger status.

| Component | Current Confidence | Missing Validation | Priority |
| --- | --- | --- | --- |
| `runtime.desktop.packaged` | Low | CPU/CUDA portable and MSI build, launch, readiness, shutdown, data-root isolation, and artifact verification. | High |
| `persistence.postgresql` | Low | Live schema/migration/repository contract against the supported PostgreSQL configuration. | Medium |
| `workflow.dataset_upload_and_preparation` | Low | Non-empty upload through image matching, unmatched confirmation, processing, and persisted metadata. | High |
| `workflow.training_and_resume` | Low | Minimal real training, checkpoint registration, cancellation, resume, and cleanup. | High |
| `workflow.dataset_validation` | Low | Successful dataset validation, artifact persistence, and rendered report review. | Medium |
| `workflow.checkpoint_evaluation` | Low | Successful compatible checkpoint evaluation and report retrieval. | Medium |
| `workflow.inference.generation_pipeline` | Medium | Provider-independent browser Generate/Cancel/provenance flow, reuse after restart, timeout, and persistence failure behavior. | High |
| `inference.model.*` quality | Low | Multi-case quality gates for all public models; CXRMate-ED currently has a failed gate. | High |
| `ui.inference.catalogue_and_sections` | Medium | Browser Generate/Cancel/edit/export against a current installed model for each relevant output-section contract. | High |
| `launcher.maintenance` | Low | Confirmation-gated KillProcesses, ClearCache, RemoveCheckpoints, RemoveAllData, and uninstall behavior on disposable fixtures. | Medium |

## Resolved / Historical Findings

These entries are retained only to prevent obsolete findings from being
mistaken for active issues. They do not change the current ledger status.

| ID | Component | Previous Finding | Resolution Evidence | Current Residual or Follow-up |
| --- | --- | --- | --- | --- |
| `HIST-001` | `inference.model.chexone` | The 2026-09-19 five-model aggregate rejected CheXOne because it expected incomplete report sections. | [aggregate run](../QA/inference_validation_runs/public-inference-models-20260919T165746Z.json) records the old failure; the current catalogue declares Findings-only, and the [2026-09-20 receipt](../QA/inference_validation/StanfordAIMI__CheXOne-0c350e6852ea08f9d9baf3b7595c1a10d4849927.json) passed with a non-empty Findings report. | Technical generation works for one fixture; `validation_status=pending` and the aggregate must be rerun. |
| `HIST-002` | `runtime.desktop.toolchain` | A settings-migration QA note recorded a pre-existing Tauri capability error during `cargo check`. | The later [2026-09-17 E2E note](../QA/e2e-validation-20260917.md) records `cargo check` passed. | Packaged artifact smoke remains validation debt under `runtime.desktop.packaged`. |
| `HIST-003` | `configuration.runtime_settings` | Application settings previously had a JSON persistence path and competing runtime authority. | [settings migration validation](../QA/settings-migration-validation-20260917.md) records the database migration, API allowlist, and rollback checks; [persistence](architecture/persistence.md) documents the one-time compatibility import. | Legacy JSON reading is intentionally bounded to the migration; revalidate only when schema/settings behavior changes. |
| `HIST-004` | `backend.ml_import_boundaries` | Prior validation identified startup import/deadlock risk around ML frameworks. | [2026-09-19 backend smoke](../QA/xreport-backend-smoke-20260919.log) reports zero `_ModuleLock`, partially initialized torch, circular Keras/PyTorch, and HTTP 500 matches. | Keep the clean-subprocess and lightweight-endpoint checks as regression gates. |

## Evidence and Ownership Boundaries

- Architecture documents describe structure, dependency direction, contracts,
  and persistence ownership. They remain the detailed technical authority.
- QA reports, test files, screenshots, logs, and receipts explain how a status
  was established. They are evidence, not substitutes for current status.
- Implementation plans describe intended work and do not change a component's
  status until implementation and evidence exist.
- This ledger is the canonical current operational summary. When detailed
  documents disagree with a current validated artifact or source contract,
  update the affected documentation and this ledger together; do not preserve
  an obsolete failure in the active issue list.
