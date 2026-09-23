# XREPORT Project Status Ledger

Last updated: 2026-09-23

This is the canonical high-level operational status catalog for the current
XREPORT checkout. It summarizes what is working, validated, partial, blocked,
unvalidated, or absent, and points to the detailed architecture and QA evidence
that supports each claim. It is a current-state index, not a development diary,
issue tracker replacement, release approval, or clinical-quality statement.

Validation baseline: `develop` started at
`481605b1b87035b8deb03edaefdbfc090f8f1b23`. The 2026-09-22 startup change set
was validated in that worktree before commit; see the dated Tier 0 summary.
Current application and validation/test revision: `37a4b78e6ca939f8a2b6cb9e29c8fa7d46a3d41e`
(`develop`). S20 passed its focused local API and service checks on this
revision; see the [S20 summary](../QA/validation_campaign/s20/summary-20260923.md).
The latest recorded hosted CI run, `35841152401`, passed on the prior source/test
revision `f6ee0a9dacf4b024cf2b8cea75c6f025c2dc7d63`; a hosted recheck of S20 is
pending. S02, S13, S14, and S15 were also revalidated on 2026-09-23; their dated
summaries and the validation campaign ledger retain their exact evidence scope.

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

Evidence under `assets/QA/` is split by retention intent. Durable campaign
summaries under `assets/QA/validation_campaign/` are tracked; large or
transient captures elsewhere may remain local and ignored. If a linked artifact
is absent in another checkout, its claim must be treated as unvalidated until
the evidence is restored or the check is rerun. The ledger never upgrades a
claim merely because source code or a unit test exists.

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

- S20 passed its focused upload API and service tests (`12 passed`) on committed
  source/test revision `37a4b78e6ca939f8a2b6cb9e29c8fa7d46a3d41e`; see the
  [2026-09-23 S20 summary](../QA/validation_campaign/s20/summary-20260923.md).
  The latest hosted CI run `35841152401` passed on prior revision `f6ee0a9`; a
  hosted recheck for the S20 revision is pending, so S00 is not yet current-green.
  On 2026-09-23 the focused S02 gate
  again rendered the current loading screen and ready workspace after two
  unhealthy health responses; see the [2026-09-23 Tier 0
  summary](../QA/validation_campaign/tier-0/summary-20260923.md) and the
  [validation campaign ledger](validation_campaign_ledger.md).
- The hosted diagnostic run exposed a test precondition: on a clean runner, the
  service factories queried `application_settings` before the subprocess had
  initialized its database. The regression now uses an isolated SQLite
  resources directory and runs migrations before concurrent service creation;
  it retains ten fresh-process repetitions and all lazy-ML assertions. No
  production code change was needed.
- The named test and complete backend unit suite passed locally on Windows
  (`135 passed` in the suite). S01 source startup and S02 rendered readiness
  were rechecked from committed code/test SHA `3f180d2`; the launcher started
  both services and the manually opened in-app browser showed the ready
  workspace, although Windows denied the launcher's automatic browser-open
  step. Hosted CI on that SHA reran backend startup/API tests, built-server
  checks, and S03 static/client gates; it did not run the Windows launcher or
  rendered startup-browser flow.
- On 2026-09-22, the changed launcher and built-frontend path passed local
  Windows checks: `run_tests.bat` completed with 149 Python tests passed and
  one skipped, client unit/E2E slices passed, the explicit production rebuild
  refreshed build state, a warm launch reused that build, and the rendered app
  reached its ready workspace. Detailed scope and remaining edge cases are in
  the [2026-09-22 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260922.md).
- Settings persistence and reset have browser and automated evidence from
  2026-09-17; responsive settings, dataset, and inference evidence was captured
  again on 2026-09-19/20.
- S10 passed on 2026-09-22 on `develop` with direct-load/refresh coverage for
  every routed surface, exact active-navigation assertions, Light/Dark/System
  theme persistence and media changes, guidance dismissal/completion/replay,
  and clean browser console/request collectors. The durable evidence is in the
  [S10 summary](../QA/validation_campaign/s10/summary-20260922.md).
- S12 passed on 2026-09-22 for generic job API list/filter/poll/completion,
  unknown-job responses, cancellation through terminal state, typed recoverable
  failures, and client transient-poll recovery. The [S12 summary](../QA/validation_campaign/s12/summary-20260922.md)
  records the test revision and the remaining Windows pytest-cache warning.
- S13 passed on 2026-09-23 using an isolated SQLite resource root. Settings and
  a synthetic inference history record with a linked report survived a backend
  restart and API readback; the database reached Alembic head `e91a4f6c2d73`.
  Incompatible-schema and migration-rollback tests passed, and the new focused
  test verified SQLite foreign-key enforcement, orphan rejection, and cascade
  deletion. The [S13 summary](../QA/validation_campaign/s13/summary-20260923.md)
  records the exact scope and the occupied-port launcher limitation.
- S14 passed on 2026-09-23 against PostgreSQL 16 in hosted CI. Application
  migrations reached Alembic head `e91a4f6c2d73`; a settings value survived
  reinitialization, concurrent first-time initializers completed through the
  advisory locks, and a refused connection did not expose credentials in the
  error or logs. The [S14 summary](../QA/validation_campaign/s14/summary-20260923.md)
  records the scenario results and initial schema-check defect fix.
- S15 passed on 2026-09-23 from application source revision
  `0ace867ea8087a112305bb9acce1770f259e2f2c`. API regressions and the rendered
  Dataset page covered disabled/enabled access, invalid-path recovery,
  empty-folder feedback, and successful selection of a folder with one image.
  This is filesystem access and selection evidence only; it does not validate
  upload, matching, processing, or persisted dataset metadata. See the
  [S15 summary](../QA/validation_campaign/s15/summary-20260923.md).
- Real single-fixture technical inference has been observed for the three
  CXRMate public models on 2026-09-19 and CheXOne on 2026-09-20. These receipts
  do not establish clinical quality or catalog validation promotion.
- CXRMate-ED has an active degraded three-case sensitivity finding. MedGemma is
  access-blocked by its gated provider terms and credential requirement.
- Full dataset, training, successful validation/evaluation, and packaged
  desktop workflows remain validation debt even where implementation and
  focused tests exist.
- The earlier red runs remain in the historical ledger; the current CI result
  supersedes their S00 status.

## Current Component Ledger

### Runtime, configuration, and platform

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `runtime.source.startup` | `PARTIAL` | Windows source warm launch, current-build reuse, lightweight Node static/API proxy, rendered readiness, database startup checks, and selected port/build-freshness cases. | [2026-09-22 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260922.md); [2026-09-21 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260921.md); [backend smoke log](../QA/xreport-backend-smoke-20260919.log) | Permission-denied termination, post-consent PID takeover, a still-bound owner, output deletion, rebuild-after-source-change, package-lock invalidation/npm-ci recovery, and edit-during-build are not yet exercised end to end. | — | 2026-09-22 | unit + integration + E2E + manual | [startup](runtime/startup.md); [system overview](architecture/system_overview.md) | Complete the remaining launcher race/failure and dependency-invalidation cases before upgrading this scoped status. |
| `runtime.desktop.packaged` | `UNVALIDATED` | Tauri CPU/CUDA runtime extraction, portable/MSI packaging, startup, shutdown, and user-data-root isolation. | [desktop packaging tests](../../app/tests/unit/test_desktop_packaging.py); [2026-09-17 E2E note](../QA/e2e-validation-20260917.md) records `cargo check` passed. | No current CPU/CUDA portable/MSI smoke report is present under `assets/QA/desktop/`. | — | 2026-09-17 | unit + build-check | [deployment](runtime/deployment.md); [runtime modes](runtime/modes.md) | Build both variants and run the documented packaged smoke checks, preserving reports under `assets/QA/desktop/`. |
| `runtime.containerized` | `NOT_IMPLEMENTED` | Containerized runtime or image-based deployment. | [runtime modes](runtime/modes.md) explicitly records this mode as absent. | No container build, image, or deployment contract exists. | — | — | None | [runtime modes](runtime/modes.md); [deployment](runtime/deployment.md) | Define a supported container contract only if deployment scope expands. |
| `configuration.runtime_settings` | `VALIDATED` | Database-backed public settings: seed, filesystem access, polling interval, and inference timeout, including save, reload, reset, and allowlisting. | [settings validation](../QA/settings-migration-validation-20260917.md); [settings E2E](../../app/tests/e2e/test_settings_api.py); [settings page E2E](../../app/tests/e2e/test_angular_ui.py) | Hidden infrastructure, credential, and static model-policy values remain intentionally unavailable to the Settings API. | — | 2026-09-17 | unit + integration + E2E + manual | [configuration](runtime/configuration.md); [persistence](architecture/persistence.md); [UI experience](ui/experience.md) | Revalidate save/reset and migration behavior after settings or schema changes. |
| `security.authentication` | `NOT_IMPLEMENTED` | API authentication and authorization. | [execution and data flow](architecture/execution_and_data_flow.md) states that no auth layer is implemented. | Current security boundary is trusted local use; external exposure is not covered. | — | — | None | [execution and data flow](architecture/execution_and_data_flow.md); [runtime modes](runtime/modes.md) | Define and validate a threat model before supporting non-local deployment. |

### Backend, jobs, and persistence

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `backend.api.contracts` | `VALIDATED` | Tested health, model catalogue, dataset status/name, checkpoint, settings, and inference-history list/detail/filter/error surfaces plus typed error behavior. | [2026-09-22 reports validation](../QA/validation_campaign/reports-history-validation-20260922.md); [backend API tests](../../app/tests/e2e/test_inference_api.py); [OpenAPI tests](../../app/tests/unit/test_openapi_schema.py) | This status covers the exercised surface, not every endpoint or every long-running happy path. | — | 2026-09-22 | unit + integration + E2E | [backend API](architecture/backend_api.md); [system overview](architecture/system_overview.md) | Extend endpoint-level E2E coverage when a route or response contract changes. |
| `backend.jobs.lifecycle` | `VALIDATED` | Generic job start, list/filter, poll, completion, cancellation, unknown-job responses, typed terminal failure, and persistence-failure semantics. | [2026-09-22 S12 summary](../QA/validation_campaign/s12/summary-20260922.md); [API lifecycle tests](../../app/tests/integration/test_job_lifecycle_api.py); [job failure tests](../../app/tests/unit/test_job_failure_semantics.py); [job cancellation tests](../../app/tests/unit/test_job_cancellation_semantics.py) | S12 uses deterministic in-process runners; feature-specific heavy workflows remain outside this component scope. | — | 2026-09-22 | unit + integration | [execution and data flow](architecture/execution_and_data_flow.md); [backend API](architecture/backend_api.md) | Revalidate the affected job path after changes to a feature service or polling contract. |
| `backend.ml_import_boundaries` | `VALIDATED` | Lightweight service imports and concurrent service construction keep the listed Keras/PyTorch/Transformers/provider modules unloaded across ten fresh subprocess runs with an initialized isolated database. | [2026-09-22 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260922.md); [import-boundary tests](../../app/tests/unit/test_ml_import_boundaries.py); hosted CI run `35748390899` | The assertion covers the named import boundary and service factories, not every endpoint or later ML job. | — | 2026-09-22 | unit + hosted CI | [execution and data flow](architecture/execution_and_data_flow.md); [troubleshooting](operations/troubleshooting.md) | Revalidate after changes to service initialization or optional-ML import boundaries. |
| `persistence.sqlite_migrations` | `VALIDATED` | SQLite startup/initialization reaches the checked-in Alembic head `e91a4f6c2d73`, persists application settings across restart, and fails closed for incompatible schemas and interrupted migrations. | [2026-09-23 S13 summary](../QA/validation_campaign/s13/summary-20260923.md); [database initialization tests](../../app/tests/unit/test_database_initialization.py); [2026-09-22 reports validation](../QA/validation_campaign/reports-history-validation-20260922.md) | Existing unversioned or incompatible schemas intentionally fail closed. | — | 2026-09-23 | unit + integration + manual | [persistence](architecture/persistence.md); [architecture review](architecture/architecture_review.md) | Recheck migration upgrade and rollback safety for every new revision. |
| `persistence.inference_history` | `VALIDATED` | Durable session list/detail, generated-versus-edited report text, atomic updates, restart persistence of a synthetic run with linked report, and cascade deletion using the existing `InferenceRun` and `InferenceReport` entities. | [2026-09-23 S13 summary](../QA/validation_campaign/s13/summary-20260923.md); [2026-09-22 reports validation](../QA/validation_campaign/reports-history-validation-20260922.md); [repository persistence tests](../../app/tests/unit/test_repository_persistence.py); [backend API tests](../../app/tests/e2e/test_inference_api.py) | No current issue recorded for the scoped CRUD contract. | — | 2026-09-23 | unit + integration + E2E + manual | [persistence](architecture/persistence.md); [backend API](architecture/backend_api.md) | Revalidate the history contract after schema or API changes. |
| `persistence.postgresql` | `VALIDATED` | PostgreSQL 16 application migrations reach Alembic head; schema contract, settings persistence across reinitialization, concurrent initialization, advisory locks, and sanitized connection failure pass. | [2026-09-23 S14 summary](../QA/validation_campaign/s14/summary-20260923.md); [PostgreSQL contract tests](../../app/tests/integration/test_persistence_contract.py); [CI run 35841152401](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35841152401) | Scope is the exercised PostgreSQL 16 contract; it does not establish behavior for every production database configuration. | — | 2026-09-23 | unit + integration + hosted CI | [persistence](architecture/persistence.md); [deployment](runtime/deployment.md) | Revalidate after migration, connection, or database-initialization changes. |
| `persistence.checkpoint_registry` | `PARTIAL` | Database-owned checkpoint identity, complete-artifact registration, listing, and safe deletion. | [backend smoke](../QA/xreport-backend-smoke-20260919.log); [checkpoint/deletion tests](../../app/tests/e2e/test_training_api.py) | Four incomplete `e2e_delete_*` registrations produce warnings while the listing endpoint still returns HTTP 200; see `ISSUE-003`. | — | 2026-09-19 | unit + integration + manual | [persistence](architecture/persistence.md); [backend API](architecture/backend_api.md) | Reconcile incomplete fixture registrations and rerun startup plus checkpoint listing. |

### Domain workflows

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `workflow.dataset_upload_and_preparation` | `UNVALIDATED` | Non-empty upload, image matching, unmatched-row confirmation, processing, persistence, and downstream-ready dataset. | [upload E2E tests](../../app/tests/e2e/test_upload_api.py); [image scanning tests](../../app/tests/unit/test_preparation_image_scanning.py); [S15 filesystem selection evidence](../QA/validation_campaign/s15/summary-20260923.md); [S20 upload evidence](../QA/validation_campaign/s20/summary-20260923.md) | S15 validates filesystem access/path selection and S20 validates parsing/identity only; no complete upload-to-persistence happy path is evidenced. | — | 2026-09-23 | focused upload API + unit + rendered selection subflow | [workflows](operations/workflows.md); [backend API](architecture/backend_api.md) | Run a disposable non-empty dataset through load, partial-confirmation, processing, and persisted metadata checks. |
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
| `ui.inference.catalogue_and_sections` | `VALIDATED` | Rendered inference route, model-card grid, banner behavior, responsive layout, model-declared report sections, and desktop catalogue/details geometry. | [2026-09-22 reports validation](../QA/validation_campaign/reports-history-validation-20260922.md); [2026-09-19 validation note](../QA/xreport-validation-20260919.md); [inference screenshot](../QA/xreport-inference-narrow-dark.png); [CheXOne UI contract test](../../app/tests/e2e/test_angular_ui.py) | Live browser Generate/Cancel/edit/export coverage is not established for every provider. | — | 2026-09-22 | E2E + manual | [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Add a rendered Generate/Cancel/provenance flow against a current installed model. |
| `ui.reports.history` | `VALIDATED` | Reports navigation, filtered/paginated history cards, persisted detail metadata, section-aware draft editor, explicit empty/loading/error states, and edit/reload/delete behavior. | [2026-09-22 reports validation](../QA/validation_campaign/reports-history-validation-20260922.md); [Reports route E2E](../../app/tests/e2e/test_angular_ui.py); [section helper unit tests](../../app/client/src/app/common/report-sections.spec.ts) | The CRUD browser evidence uses a deterministic disposable fixture; clinical review quality and provider-wide generation remain outside this surface. | — | 2026-09-22 | unit + E2E + manual | [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Revalidate responsive history behavior after route or editor changes. |
| `ui.startup_gate` | `VALIDATED` | Shell-level startup surface, serialized `/api/health` readiness polling, route suppression before readiness, recovery, ready transition, responsive composition, and current built-bundle rendering. | [2026-09-23 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260923.md); [current startup screenshots](../QA/validation_campaign/tier-0/s02-20260923/); [2026-09-22 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260922.md); [startup E2E](../../app/tests/e2e/test_angular_ui.py); [frontend unit suite](../../app/client/src/app/services/startup-readiness.service.spec.ts) | Explicit slow/unavailable/retry timing and ordinary post-ready feature-error manual evidence remain open; packaged CPU/CUDA startup is not covered. | — | 2026-09-23 | unit + E2E + manual | [startup](runtime/startup.md); [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Recheck explicit recovery states and packaged CPU/CUDA startup after the next desktop validation run. |
| `ui.shell.routing_theme_guidance` | `VALIDATED` | Routed shell surfaces, root redirect, direct-load/refresh behavior, exact active navigation for parameterized child routes, Light/Dark/System theme resolution and persistence, versioned guidance dismissal/completion, and manual replay. | [2026-09-22 S10 summary](../QA/validation_campaign/s10/summary-20260922.md); [S10 E2E coverage](../../app/tests/e2e/test_angular_ui.py); [route contract](../../app/client/src/app/app.routes.ts) | The S10 browser gate used disposable report and browser-scoped validation fixtures; complete dataset/training workflows remain outside this status. | — | 2026-09-22 | E2E + manual | [UI experience](ui/experience.md); [UI patterns](ui/components_and_patterns.md) | Revalidate after route, shell, theme, or guidance contract changes. |
| `ui.dataset_and_training_surfaces` | `WORKING` | Dataset and Training route rendering, navigation, and responsive surface. | [S10 route matrix](../QA/validation_campaign/s10/summary-20260922.md); [route-render E2E](../../app/tests/e2e/test_angular_ui.py); [dataset screenshot](../QA/xreport-dataset-narrow-dark.png) | Route rendering is evidenced, but complete dataset and training workflows remain unvalidated. | — | 2026-09-22 | E2E + manual | [UI patterns](ui/components_and_patterns.md); [workflows](operations/workflows.md) | Pair the surface checks with live dataset and minimal-training workflow evidence. |
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
| `ISSUE-004` | `test.infrastructure.windows_cache` | `LOW` | Test/E2E runs pass but emit permission warnings for pre-existing protected pytest/cache directories. | Results remain usable, but cleanup and reproducibility are less clear on Windows. | [settings QA](../QA/settings-migration-validation-20260917.md), [E2E QA](../QA/e2e-validation-20260917.md), and [S12 summary](../QA/validation_campaign/s12/summary-20260922.md) record current warnings. | Protected pre-existing cache ACLs, as documented by the validation runs. | Filesystem permission/ownership of the legacy cache paths. | `OPEN` — keep active caches under `runtimes/cache` and report any protected legacy paths explicitly. | Rerun the relevant tests and confirm no unexpected cache-permission warnings remain. | [testing rules](coding/testing_and_quality.md); [startup](runtime/startup.md) |

## Validation Debt

Validation debt is not a defect list. It identifies important behavior whose
current confidence is too low or too narrow to support a stronger status.

| Component | Current Confidence | Missing Validation | Priority |
| --- | --- | --- | --- |
| `runtime.desktop.packaged` | Low | CPU/CUDA portable and MSI build, launch, readiness, shutdown, data-root isolation, and artifact verification. | High |
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
| `HIST-005` | `validation.tier0.s00` | Hosted CI on the original test revision failed `test_concurrent_service_initialization_keeps_ml_imports_lazy`; the test relied on ambient database state and hid child stderr. | Diagnostic run `35743413087` exposed `no such table: application_settings`; current revision `3f180d229278aff379358a9f8d3eddf3b07dee5c` initializes an isolated test database before service construction and passed hosted CI run `35748390899`. | No production race was established; keep the repeated clean-process regression and current CI gate. |

## Evidence and Ownership Boundaries

- Architecture documents describe structure, dependency direction, contracts,
  and persistence ownership. They remain the detailed technical authority.
- QA reports, test files, screenshots, logs, and receipts explain how a status
  was established. They are evidence, not substitutes for current status.
- [`validation_campaign_ledger.md`](validation_campaign_ledger.md) is the
  subordinate slice authority for campaign order and remaining gaps; its
  durable Tier 0 execution summary is retained under
  `assets/QA/validation_campaign/`.
- Implementation plans describe intended work and do not change a component's
  status until implementation and evidence exist.
- This ledger is the canonical current operational summary. When detailed
  documents disagree with a current validated artifact or source contract,
  update the affected documentation and this ledger together; do not preserve
  an obsolete failure in the active issue list.
