# XREPORT Project Status Ledger

Last updated: 2026-10-18

This is the canonical high-level operational status catalog for the current
XREPORT checkout. It summarizes what is working, validated, partial, blocked,
unvalidated, or absent, and points to the detailed architecture and QA evidence
that supports each claim. It is a current-state index, not a development diary,
issue tracker replacement, release approval, or clinical-quality statement.

Validation baseline: the current release source is `a8d75a0c0d5f02a29c85e9ac58f4837fc3081449`
(v3.2.0, published 2026-10-18). Slice-level baselines, per-run revisions, and
campaign history are catalogued in [`validation_campaign_ledger.md`](validation_campaign_ledger.md).
Evidence paths for campaigns consolidated before 2026-10-18 were removed from
the working tree on that date; those records remain recoverable from Git
history, and the cells below link only records present in the current tree.

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

Tracked evidence is limited to compact durable records: per-slice summaries
under `assets/QA/validation_campaign/`, the approved release manifest, and the
desktop release-gate receipts under `assets/QA/desktop/`. When a linked artifact
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

- On 2026-10-18, release workflow run `39103683278` published v3.2.0 from
  source `a8d75a0c0d5f02a29c85e9ac58f4837fc3081449`. The tag points to that
  source, all eight published asset digests match the approved manifest, and
  the workflow passed CPU/CUDA artifact revalidation and portable smoke. See
  the [release record](../QA/validation_campaign/release-3.2.0-20261017/summary-20261017.md)
  and the [approved manifest](../QA/validation_campaign/release-validation/approved-release-manifest.json).
- The accepted packages come from run `39056753392` (the earlier run
  `39028661915` was invalidated by `ISSUE-007`/`ISSUE-008` and superseded).
  CPU/CUDA portable and MSI packages passed native routes, Training
  submit/cancel, active-work close/reopen, and pinned synthetic inference; CPU
  resolved `cpu` with CUDA unavailable, CUDA resolved `cuda:0`. The final CPU
  MSI uninstall on 2026-10-18 returned exit code `0` with no residual
  registration, install tree, shortcut, process, or listener.
- Hosted CI is green on the validation revision
  `7bf8402f90bdbf2181f13190782b67ba51d445d9` (run `36422997859`) and on the
  later `064278f8` (run `36586991512`). The full live-service Python suite
  passed `235` with `3` documented PostgreSQL skips; the official Windows
  `app/tests/run_tests.bat` runner passed (Python `191`, frontend unit `40`,
  Angular E2E `14`). The Angular production builder crashed intermittently
  with Windows `0xC0000005` in later local runs, so no new local
  production-build PASS is claimed beyond hosted CI.
- Validation state: the desktop package binding is current for the published
  v3.2.0 artifacts; `S42` and `S51` remain `PARTIAL` in the campaign ledger.
  No GPU-less Windows host is available; CUDA fallback remains untested by user
  direction and is not a primary release blocker. `ISSUE-002` remains
  `BLOCKED` on authorized gated access; `ISSUE-004` remains open for legacy
  protected cache paths. `ISSUE-007` and `ISSUE-008` are fixed in the exact
  v3.2.0 packages.

## Current Component Ledger

> Evidence cells cite surviving tracked records and (where noted) the validating
> action and date; per-run captures from earlier campaigns remain in Git
> history per the evidence policy above.

### Runtime, configuration, and platform

| Component | Status | Scope | Evidence | Known Issues | Last Validated | Validation Level | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `runtime.source.startup` | `PARTIAL` | Windows source warm launch, current-build reuse, lightweight Node static/API proxy, rendered readiness, isolated SQLite startup and restart, deterministic incompatible-schema fail-closed startup, selected port/build-freshness cases, and current `data/` root initialization/override. | [runtime startup](runtime/startup.md); [testing rules](coding/testing_and_quality.md); official-launcher rechecks on disposable roots (2026-09-28). | Permission-denied termination, post-consent PID takeover, a still-bound owner, output deletion, rebuild-after-source-change, package-lock/npm-ci recovery, and edit-during-build are not exercised end to end; a final `KillProcesses` run required exact-PID cleanup after an environment-level process-inspection `Access denied`. Packaged CPU/CUDA data-root behavior is a separate desktop slice. | 2026-09-29 | unit + integration + E2E + manual | Exercise the interactive ownership/race and rebuild-invalidation paths. |
| `runtime.desktop.packaged` | `PARTIAL` | Tauri CPU/CUDA runtime extraction, portable/MSI packaging, packaged startup and shutdown, user-data-root isolation, native CPU/CUDA interaction receipts, full MSI install→launch→graceful-close→uninstall lifecycles, and packaged CPU/CUDA inference with persisted device provenance for the published v3.2.0 packages. | [release record](../QA/validation_campaign/release-3.2.0-20261017/summary-20261017.md); [approved manifest](../QA/validation_campaign/release-validation/approved-release-manifest.json); [CPU runtime audit](../QA/desktop/runtime-cpu-3.2.0.json); [CUDA runtime audit](../QA/desktop/runtime-cuda-3.2.0.json); [CPU verification](../QA/desktop/verification-cpu-3.2.0.json); [CUDA verification](../QA/desktop/verification-cuda-3.2.0.json); [CPU smoke](../QA/desktop/smoke-cpu-3.2.0.json); [CUDA smoke](../QA/desktop/smoke-cuda-3.2.0.json); [desktop packaging tests](../../app/tests/unit/test_desktop_packaging.py) | Genuine no-GPU fallback is untested by user direction; representative performance and cross-variant benchmark equivalence remain open. | 2026-10-18 | unit + official launcher + package verification/smoke + native UI receipts + packaged real inference + elevated MSI lifecycle receipts | Run a genuine no-GPU lane if hardware becomes available; revalidate affected variants on each release. |
| `runtime.containerized` | `NOT_IMPLEMENTED` | Containerized runtime or image-based deployment. | [runtime modes](runtime/modes.md) records this mode as absent; [deployment](runtime/deployment.md). | No container build, image, or deployment contract exists. | — | None | Define a supported container contract only if deployment scope expands. |
| `configuration.runtime_settings` | `VALIDATED` | Database-backed public settings: seed, filesystem access, polling interval, and inference timeout, including save, reload, reset, and allowlisting. | [settings E2E](../../app/tests/e2e/test_settings_api.py); [settings page E2E](../../app/tests/e2e/test_angular_ui.py); live GET/PATCH/reset receipt (2026-10-01, archived); [configuration](runtime/configuration.md). | Hidden infrastructure, credential, and static model-policy values remain intentionally unavailable to the Settings API. | 2026-10-01 | unit + integration + E2E + manual | Revalidate save/reset and migration behavior after settings or schema changes. |
| `security.authentication` | `NOT_IMPLEMENTED` | API authentication and authorization. | [execution and data flow](architecture/execution_and_data_flow.md) states that no auth layer is implemented; [runtime modes](runtime/modes.md). | Current security boundary is trusted local use; external exposure is not covered. | — | None | Define and validate a threat model before supporting non-local deployment. |

### Backend, jobs, and persistence

| Component | Status | Scope | Evidence | Known Issues | Last Validated | Validation Level | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `backend.api.contracts` | `VALIDATED` | Tested health, model catalogue, dataset status/name, checkpoint, settings, and inference-history list/detail/filter/error surfaces plus typed error behavior. | [backend API tests](../../app/tests/e2e/test_inference_api.py); [OpenAPI tests](../../app/tests/unit/test_openapi_schema.py); [backend API](architecture/backend_api.md). | Covers the exercised surface, not every endpoint or every long-running happy path. | 2026-09-22 | unit + integration + E2E | Extend endpoint-level E2E coverage when a route or response contract changes. |
| `backend.jobs.lifecycle` | `VALIDATED` | Generic job start, list/filter, poll, completion, cancellation, unknown-job responses, typed terminal failure, and persistence-failure semantics. | [API lifecycle tests](../../app/tests/integration/test_job_lifecycle_api.py); [failure tests](../../app/tests/unit/test_job_failure_semantics.py); [cancellation tests](../../app/tests/unit/test_job_cancellation_semantics.py). | S12 used deterministic in-process runners; feature-specific heavy workflows remain outside this scope. | 2026-09-22 | unit + integration | Revalidate the affected job path after feature or polling-contract changes. |
| `backend.ml_import_boundaries` | `VALIDATED` | Lightweight service imports and concurrent service construction keep the listed Keras/PyTorch/Transformers/provider modules unloaded across ten fresh subprocess runs with an initialized isolated database. | [import-boundary tests](../../app/tests/unit/test_ml_import_boundaries.py); hosted CI run `35748390899`. | Covers the named import boundary and service factories, not every endpoint or ML job. | 2026-09-22 | unit + hosted CI | Revalidate after service-initialization or optional-ML boundary changes. |
| `persistence.sqlite_migrations` | `VALIDATED` | SQLite startup/initialization reaches the checked-in Alembic head `e91a4f6c2d73`, persists application settings and an inference report across restart, fails closed for deterministic incompatible schema drift, and supports backup restoration with clean integrity. | [database initialization tests](../../app/tests/unit/test_database_initialization.py); [persistence](architecture/persistence.md). | Existing unversioned or incompatible schemas intentionally fail closed. Crash recovery, partial migration interruption, filesystem corruption, and PostgreSQL recovery remain outside S50. | 2026-09-28 | unit + integration + manual rendered browser | Recheck migration upgrade and rollback safety for every new revision. |
| `persistence.inference_history` | `VALIDATED` | Durable session list/detail, generated-versus-edited report text, atomic updates, restart persistence of a synthetic run with linked report, and cascade deletion using the existing entities. | [repository persistence tests](../../app/tests/unit/test_repository_persistence.py); [backend API tests](../../app/tests/e2e/test_inference_api.py). | No current issue recorded for the scoped CRUD contract. | 2026-09-23 | unit + integration + E2E + manual | Revalidate the history contract after schema or API changes. |
| `persistence.postgresql` | `VALIDATED` | PostgreSQL 16 application migrations reach Alembic head; schema contract, settings persistence across reinitialization, concurrent initialization, advisory locks, and sanitized connection failure pass. | [PostgreSQL contract tests](../../app/tests/integration/test_persistence_contract.py); [CI run 35841152401](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35841152401). | Scope is the exercised PostgreSQL 16 contract; not every production configuration is covered. | 2026-09-23 | unit + integration + hosted CI | Revalidate after migration, connection, or initialization changes. |
| `persistence.checkpoint_registry` | `VALIDATED` | Database-owned checkpoint identity, complete-artifact registration, metadata/history listing, safe and referenced deletion, and artifact/registry cleanup. | [checkpoint/deletion tests](../../app/tests/e2e/test_training_api.py); [persistence](architecture/persistence.md). | Synthetic technical checkpoint only. `ISSUE-003` was traced to an incomplete test fixture and corrected. | 2026-09-23 | unit + API E2E + manual | Revalidate after artifact-format, reference, or deletion-contract changes. |

### Domain workflows

| Component | Status | Scope | Evidence | Known Issues | Last Validated | Validation Level | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `workflow.dataset_upload_and_preparation` | `VALIDATED` | Synthetic non-empty upload through image matching, partial confirmation, processing, persistence, and downstream-ready processed dataset. | [image scanning tests](../../app/tests/unit/test_preparation_image_scanning.py); [dataset API tests](../../app/tests/e2e/test_dataset_workflow_api.py); [dataset page test](../../app/tests/e2e/test_dataset_workflow_ui.py). | Evidence uses an eight-row synthetic corpus; large/real-world data, packaged operation, and clinical quality remain outside. Source deletion returns 409 while processed dependents exist; processed-first deletion is verified in the browser. | 2026-09-28 | unit + API contract + browser E2E + live processing + restart persistence | Broaden with representative fixtures and an explicit scale/quality scope. |
| `workflow.training_and_resume` | `VALIDATED` | One-epoch real CUDA training on a tiny synthetic dataset, checkpoint creation/registration, managed cancellation after progress, resume with persisted history advancing by one epoch, atomic same-type admission, lifecycle phase/first-batch reporting, repeated clean-restart source matrices, and packaged CPU active close/reopen. | [training API tests](../../app/tests/e2e/test_training_api.py); [training worker tests](../../app/tests/unit/test_training_worker_diagnostics.py); [watchdog tests](../../app/tests/unit/test_training_startup_watchdog.py). | Synthetic eight-row scope; CPU-only packaged fallback, expanded scale, native-form submission semantics, and representative data remain outside. | 2026-10-01 | unit + API E2E + live CUDA jobs + clean-restart matrices + packaged receipt | Broaden only with representative fixtures and explicit scope. |
| `workflow.dataset_validation` | `VALIDATED` | Full-dataset validation of an eight-row synthetic dataset; text, image, and pixel metrics completed; metric values and report persisted and retrieved; completion view showed saved metadata; concurrent-triple repeats kept stable per-job report identity. | [validation contract tests](../../app/tests/unit/test_validation_contract_metrics.py); [dataset page regression test](../../app/client/src/app/pages/dataset.page.spec.ts). | Synthetic technical evidence only; no representative-data, scale, or clinical-quality claim. The report `artifacts` map was empty, so separate artifact-file generation was not exercised. | 2026-09-29 | unit + API integration + repeated live CUDA jobs + manual rendered review + production build | Revalidate metric/report changes; validate file-artifact generation if introduced. |
| `workflow.checkpoint_evaluation` | `VALIDATED` | Compatible checkpoint evaluation, associated dataset resolution, metrics, and report persistence on an eight-row synthetic CUDA workflow; repeated runs kept expected persisted checkpoint/report identity and overlapped safely with dataset processing. | [evaluation tests](../../app/tests/unit/test_evaluation.py); [validation job tests](../../app/tests/unit/test_validation_job_semantics.py). | Synthetic and one epoch only; no representative-data, clinical-quality, scale, artifact-file, or packaged-release claim. | 2026-09-29 | unit + repeated live CUDA jobs + API + manual rendered review | Broaden only with representative fixtures and an explicit quality/performance scope. |
| `workflow.inference.generation_pipeline` | `VALIDATED` | One exact-pinned public-model install in an isolated runtime, real study generation, rendered Generate/poll/cancel/retry/review, provenance, report history, edit/save/reload, copy/export, restart/reuse lifecycle, and supported canonical-resource repair. | [provider tests](../../app/tests/unit/test_huggingface_provider.py); [inference API E2E](../../app/tests/e2e/test_inference_api.py). | Limited to CXRMate Multi and one public image; the canonical local model resource matches the pinned S31 manifest after repair, and one disposable Generate call recorded verified integrity. `validation_status` remains `pending`; no quality, clinical, or multi-provider claim. | 2026-10-01 | integration + client unit + manual browser + restart/runtime + supported model repair + disposable inference | Maintain manifest/integrity on model changes; cover other providers, timeout/persistence-failure behavior, and multi-case quality separately. |

### Model providers

| Component | Status | Scope | Evidence | Known Issues | Last Validated | Validation Level | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `inference.catalogue` | `VALIDATED` | Four currently declared public entries, exact revisions, adapters, access/readiness state, and model-declared output sections, with public/custom identity separation. | [catalogue unit tests](../../app/tests/unit/test_inference_model_catalog.py); [API E2E](../../app/tests/e2e/test_inference_api.py); [model configuration](../../app/server/configurations/inference_models.py). | CXRMate-ED was removed after no corrected qualification gate passed. Catalogue cardinality is not a product invariant; live custom listing remains unobserved. | 2026-09-29 | unit + API E2E + rendered UI + bounded model diagnostic | Revalidate after manifest/catalog changes and qualify any replacement independently. |
| `inference.model.cxrmate-multi` | `PARTIAL` | Pinned CXRMate Multi TF revision install, one-image Findings/Impression generation, readiness promotion, restart reuse, delete lifecycle, one packaged CPU and CUDA inference with persisted device provenance, and repaired canonical integrity. | [provider tests](../../app/tests/unit/test_huggingface_provider.py); [release record](../QA/validation_campaign/release-3.2.0-20261017/summary-20261017.md) (packaged inference scope). | The supported repair workflow restored all eight canonical files to the pinned manifest and one disposable Generate call completed with non-empty Findings and Impression output and verified integrity. `validation_status` remains `pending`; no quality or clinical claim. | 2026-10-01 | isolated integration + packaged real inference + manual browser + manifest comparison + supported repair | Keep manifest verification on model changes and qualify broader quality separately. |
| `inference.model.cxrmate-ed` | `DEPRECATED` | Retired CXRMate-ED entry and adapter after the bounded upstream-contract rescue failed its independent-case responsiveness gate. | [catalogue/API regression](../../app/tests/e2e/test_inference_api.py). | The adapter produced one identical full report across three distinct fixtures under the published contract; the reference process was blocked by the unavailable pinned runtime, so no quality PASS is claimed. Historical reports, provenance, and downloaded files are preserved. | 2026-09-29 | bounded technical diagnostic + unit/API/UI regression | Do not restore or replace without an independent three-case qualification receipt. |
| `inference.model.chexone` | `VALIDATED` | Pinned CheXOne Findings-only report generation and declared UI contract. | [CheXOne contract tests](../../app/tests/e2e/test_inference_api.py); three-case technical receipt (2026-09-26, archived). | Exact revision `0c350e6852ea08f9d9baf3b7595c1a10d4849927` passed the independent three-case technical run, reuse sentinel, conservative review, and a rendered catalogue check. Research-only Findings drafting; no diagnostic-accuracy claim. | 2026-09-26 | isolated integration + manual review + browser catalogue | Extend with labelled-reference quality studies before any clinical-performance claim. |
| `inference.model.cxrmate-2` | `VALIDATED` | Pinned CXRMate-2 Findings/Impression generation on a high-demand local runtime. | Three-case technical receipt (2026-09-26, archived); [local inference models](runtime/local_inference_models.md). | Exact revision `aa8e2d16470e20671acf049687b4707c9bf2f2b5` passed the independent three-case technical run, very-high resource/reuse sentinel, conservative review, and a rendered catalogue check. Research-only drafting; no diagnostic-accuracy claim. | 2026-09-26 | isolated integration + manual review + browser catalogue | Extend with labelled-reference quality studies before any clinical-performance claim. |
| `inference.model.medgemma` | `BLOCKED` | Gated MedGemma installation and local generation. The gated-provider behavior contract (S37) is separately `PASS`: catalogue `access_policy=gated`/`not_installed`, backend 409 rejection before job start, and a disabled `Access required` UI. | S37 live receipt (2026-10-01, archived); [gated access](runtime/local_inference_models.md); [configuration](runtime/configuration.md). | The live catalogue reports `access_policy=gated` and `not_installed`; `settings/.env` `HF_TOKEN` is empty and no installation was attempted. The stored credential store was not probed. See `ISSUE-002`. | 2026-10-01 | live API + rendered UI + unit/regression | After access is authorized, download the exact pinned revision and capture a real inference receipt. |

### User interface and test infrastructure

| Component | Status | Scope | Evidence | Known Issues | Last Validated | Validation Level | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ui.settings` | `VALIDATED` | Settings route navigation, four public controls, save/reload/reset, explicit states, responsive layout, and rendered restart readback of the database-backed seed. | [settings route E2E](../../app/tests/e2e/test_angular_ui.py); [UI experience](ui/experience.md). | No current issue recorded for the exercised settings surface. | 2026-09-28 | E2E + manual rendered browser | Revalidate persistence and responsive states after route, contract, or token changes. |
| `ui.inference.catalogue_and_sections` | `VALIDATED` | Rendered inference route, model-card grid, banner behavior, responsive layout, model-declared report sections, model-dependent context controls, profile changes, desktop catalogue/details geometry, and a bounded internally scrollable catalogue for large checkpoint counts. | [large-catalogue regression](../../app/client/src/app/pages/job-cancellation.pages.spec.ts); [CheXOne UI contract test](../../app/tests/e2e/test_angular_ui.py). | Live geometry passed with the current catalogue data and the client regression rendered all `40` checkpoint cards in the scroll region. Full generation/review UI evidence covers only CXRMate Multi and one public image; spoken screen-reader observation is a non-gating follow-up. | 2026-09-29 | API E2E + client unit + production build + manual rendered review | Revalidate provider-declared sections against each relevant model's rendered flow. |
| `ui.inference.workflow` | `PARTIAL` | Rendered one-model Generate/poll/cancel/retry/review/provenance/history/edit/copy/export flow plus focused inference-route keyboard, modal, responsive, and reduced-motion checks. | [inference regression tests](../../app/client/src/app/pages/job-cancellation.pages.spec.ts); [modal focus regression test](../../app/client/src/app/components/modal-focus.directive.spec.ts). | One provider and one public image only. The 25-case matrix passed layout, modal, focus, Escape, reduced-motion, screenshot, and browser-error checks. The browser download event was not surfaced in this slice. Spoken accessibility is tracked separately and is non-gating. | 2026-09-28 | client unit + production build + manual browser E2E | Expand coverage to other model output contracts and download-event observation. |
| `ui.reports.history` | `VALIDATED` | Reports navigation, filtered/paginated history cards, persisted detail metadata, section-aware draft editor, explicit empty/loading/error states, edit/reload/delete behavior, and rendered post-restore history readback. | [Reports route E2E](../../app/tests/e2e/test_angular_ui.py); [section helper unit tests](../../app/client/src/app/common/report-sections.spec.ts). | CRUD browser evidence uses deterministic disposable fixtures; clinical review quality and provider-wide generation remain outside this surface. | 2026-09-28 | unit + E2E + manual rendered browser | Revalidate responsive history behavior after route or editor changes. |
| `ui.startup_gate` | `VALIDATED` | Shell-level startup surface, serialized `/api/health` readiness polling, route suppression before readiness, controlled slow/unavailable/retry recovery, deterministic source-backend schema-drift failure state, ready transition, ordinary post-ready feature error, responsive composition, and current built-bundle rendering. | [startup E2E](../../app/tests/e2e/test_angular_ui.py); [startup-readiness unit suite](../../app/client/src/app/services/startup-readiness.service.spec.ts). | Slow/unavailable timing uses a controlled browser clock; S50 additionally observed the real frontend against a failed source backend. Packaged CPU/CUDA startup is not covered. | 2026-09-28 | unit + E2E + manual rendered browser | Recheck packaged CPU/CUDA startup after the next desktop validation run. |
| `ui.shell.routing_theme_guidance` | `VALIDATED` | Routed shell surfaces, root redirect, direct-load/refresh behavior, exact active navigation for parameterized child routes, Light/Dark/System theme resolution and persistence, versioned guidance dismissal/completion, and manual replay. | [S10 E2E coverage](../../app/tests/e2e/test_angular_ui.py); [route contract](../../app/client/src/app/app.routes.ts). | The S10 browser gate used disposable report and browser-scoped validation fixtures; complete dataset/training workflows remain outside this status. | 2026-09-22 | E2E + manual | Revalidate after route, shell, theme, or guidance contract changes. |
| `ui.dataset_and_training_surfaces` | `PARTIAL` | Dataset and Training route rendering; dataset table source-path containment and title; rendered eight-image viewer navigation; processed dataset/checkpoint metadata; dependent-delete conflict and processed-first cleanup; evaluation entry/report review; and the 25-case direct-route responsive/reduced-motion matrix. | [dataset workflow E2E](../../app/tests/e2e/test_dataset_workflow_ui.py). | Broader dataset lifecycle, representative-data behavior, and packaged/native desktop behavior remain outside this scope; spoken accessibility is tracked separately and is non-gating. | 2026-09-28 | E2E + manual + live technical workflow | Expand only with representative fixtures or a separate packaged/accessibility scope. |
| `ui.validation.report_review` | `VALIDATED` | Successful synthetic dataset-validation and checkpoint-evaluation report completion/review, including persisted timestamps/metrics and reload persistence; missing-data error path covered by the existing E2E. | [dataset page regression test](../../app/client/src/app/pages/dataset.page.spec.ts); [missing-dataset E2E](../../app/tests/e2e/test_angular_ui.py). | Eight-row synthetic technical scope only; standalone browser screenshot was not exported. No clinical, representative-data, scale, or report-quality claim. | 2026-09-25 | client unit + API + manual rendered review | Revalidate after route/report changes; broaden only with an explicit representative-data scope. |
| `test.infrastructure.windows_cache` | `PARTIAL` | Repeatable Windows test startup and disposable-cache routing. | [testing rules](coding/testing_and_quality.md); [startup](runtime/startup.md); full-run evidence (2026-09-24/28, archived). | The focused backend and browser validations used writable disposable basetemp/cache paths because protected historical paths remain ACL-inaccessible; Git still reports access warnings outside that root. The configured runner is validated. | 2026-09-28 | focused backend + client unit + browser E2E + manual | Resolve access/ownership for the historical paths before attempting cleanup; keep using validated isolated runtime caches. |

## Open Issues

Severity is deliberately separate from functional status: a `PARTIAL` component
may have a `LOW` issue, and a `BLOCKED` component may have no software defect.
Only actionable current problems belong here.

| ID | Affected Component | Severity | Concise Description | Current Impact | Reproduction or Evidence | Blocker | Remediation Status | Required Revalidation | Related Documentation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ISSUE-002` | `inference.model.medgemma` | `MEDIUM` | The gated MedGemma entry cannot be installed or generated without authorized provider access and a supported credential. The S37 gated-provider behavior gate passed separately on 2026-10-01 without model access. | One current catalogue entry is unavailable in this environment; the gated behavior contract (catalogue state, 409 rejection, disabled UI) is validated. | S37 live receipt (2026-10-01, archived): `access_policy=gated`, `not_installed`, no `HF_TOKEN`. Not a confirmed software defect. | Authorized gated access and a supported credential. | `BLOCKED` — waiting for authorized access configuration. | Download the exact pinned revision, run real inference, and capture a receipt before changing status. | [gated access](runtime/local_inference_models.md); [configuration](runtime/configuration.md) |
| `ISSUE-004` | `test.infrastructure.windows_cache` | `LOW` | Legacy unconfigured pytest runs warned on protected cache paths. The full official `run_tests.bat` now passes through configured runtime caches without pytest cache warnings. | Test execution is validated; Git directory scans still warn when they encounter some pre-existing protected cache paths outside the configured root. | Full-run logs (2026-09-24, archived); [testing rules](coding/testing_and_quality.md). | Filesystem permission/ownership of legacy cache paths. | `OPEN` — official runner is validated; legacy protected paths remain inaccessible and were left unchanged. | Verify ownership/access and safe disposition of the historical paths before cleanup; do not remove cache residue without confirmed ownership. | [testing rules](coding/testing_and_quality.md); [startup](runtime/startup.md) |

## Resolved Issues

| ID | Affected Component | Resolution and Revalidation | Evidence |
| --- | --- | --- | --- |
| `ISSUE-001` | `inference.model.cxrmate-ed` | CXRMate-ED was retired from the active catalogue because the bounded upstream-contract diagnostic still collapsed three distinct fixtures to one full report. Backend admission now rejects any degraded public model before job start, the Angular page disables Generate, and the retired ref is rejected by the API. Historical reports, provenance, and downloaded files are preserved. | [backend regression](../../app/tests/unit/test_inference_service.py); [UI regression](../../app/client/src/app/pages/job-cancellation.pages.spec.ts); [API regression](../../app/tests/e2e/test_inference_api.py) |
| `ISSUE-003` | `persistence.checkpoint_registry` | The checkpoint E2E fixture wrote an empty session history, causing complete-artifact validation to warn; its fixture now contains one epoch of loss/validation history, and cleanup reconciles registry and artifact state. Isolated listing after the suite was empty with zero stale registrations. | [training API tests](../../app/tests/e2e/test_training_api.py) |
| `ISSUE-005` | `workflow.dataset_upload_and_preparation` | Source deletion now checks dependent processing runs within the deletion transaction and returns HTTP 409 with dependent dataset names; processed-then-source deletion finishes with clean foreign-key and SQLite integrity checks, confirmed in the rendered browser flow. | [deletion regression test](../../app/tests/unit/test_dataset_deletion.py) |
| `ISSUE-006` | `inference.model.cxrmate-multi` | The supported repair workflow restored the canonical resource to the exact S31 eight-file manifest at the pinned revision; one disposable Generate call completed with non-empty Findings and Impression output and the active metadata records `integrity=verified` (2026-10-01). | [provider tests](../../app/tests/unit/test_huggingface_provider.py); repair note (2026-10-01, archived) |
| `ISSUE-007` | `ui.training.wizard` | The packaged New Training Wizard used stale CSS selectors, leaving the modal header and controls misaligned in the invalidated run `39028661915`. Fixed in source `a8d75a0`; all four native wizard captures in the exact CPU/CUDA portable and MSI v3.2.0 packages show aligned modal content. No further issue-specific validation remains. | [release record](../QA/validation_campaign/release-3.2.0-20261017/summary-20261017.md) |
| `ISSUE-008` | `ui.training.cancellation` | A successful `cancelled` terminal state was logged with an unknown-error suffix and surfaced through the error signal in the invalidated run `39028661915`. Fixed in source `a8d75a0`; native receipts show terminal `cancelled` with normal feedback and no false error in the exact v3.2.0 packages. No further issue-specific validation remains. | [release record](../QA/validation_campaign/release-3.2.0-20261017/summary-20261017.md); [client regression spec](../../app/client/src/app/pages/job-cancellation.pages.spec.ts) |

## Validation Debt

Validation debt is not a defect list. It identifies important behavior whose
current confidence is too low or too narrow to support a stronger status.

| Component | Current Confidence | Missing Validation | Priority |
| --- | --- | --- | --- |
| `runtime.desktop.packaged` | Medium | Package binding, artifact verification, native interaction, and the MSI lifecycles are validated for the v3.2.0 technical scope. Genuine no-GPU fallback, representative performance, and cross-variant benchmark equivalence remain unvalidated. | High |
| `workflow.checkpoint_evaluation` | Medium | Representative-data, scale, separate artifact-file, and packaged-release evaluation evidence beyond the passed eight-row synthetic workflow. | Medium |
| `workflow.inference.generation_pipeline` | Medium | Other-provider browser flows, timeout and persistence-failure behavior, broader reuse conditions, and multi-case quality. | High |
| `inference.model.*` quality | Low | Broader labelled-reference and representative multi-case quality studies remain outstanding for all public models; CXRMate-ED is retired after its failed S35 gate, while CheXOne and CXRMate-2 passed only their conservative technical/output review. | High |
| `ui.inference.workflow` | Medium | Other-provider output contracts, browser download-event observation, and provider-wide rendered acceptance remain; spoken accessibility is a separate non-gating follow-up. | High |
| `launcher.maintenance` | Medium | Disposable cleanup paths, source and packaged CPU/CUDA process ownership, dynamic-listener termination, canonical desktop-release cleanup, idempotent repeat, and user-data preservation are evidenced. Remaining uncertainty is limited to unrelated source-startup ownership/race/failure boundaries; S43 does not claim native WebView or no-GPU behavior. | Medium |

## Resolved / Historical Findings

These entries are retained only to prevent obsolete findings from being
mistaken for active issues. They do not change the current ledger status.

| ID | Component | Previous Finding | Resolution Evidence | Current Residual or Follow-up |
| --- | --- | --- | --- | --- |
| `HIST-001` | `inference.model.chexone` | The 2026-09-19 five-model aggregate rejected CheXOne because it expected incomplete report sections. | The 2026-09-26 three-case receipt passed with non-empty Findings-only output and reload/reuse evidence. | The old aggregate limitation is superseded for the current pinned revision: `validation_status=passed` after the technical and conservative review gates, while broader labelled-reference quality remains unmeasured. |
| `HIST-002` | `runtime.desktop.toolchain` | A settings-migration QA note recorded a pre-existing Tauri capability error during `cargo check`. | A later 2026-09-17 check recorded `cargo check` passing. | CUDA packaging and native WebView coverage remain validation debt under `runtime.desktop.packaged`. |
| `HIST-003` | `configuration.runtime_settings` | Application settings previously had a JSON persistence path and competing runtime authority. | The settings-migration validation recorded the database migration, API allowlist, and rollback checks (2026-09-17); [persistence](architecture/persistence.md) documents the one-time compatibility import. | Legacy JSON reading is intentionally bounded to the migration; revalidate only when schema/settings behavior changes. |
| `HIST-004` | `backend.ml_import_boundaries` | Prior validation identified startup import/deadlock risk around ML frameworks. | The 2026-09-19 backend smoke reported zero `_ModuleLock`, partially initialized torch, circular Keras/PyTorch, and HTTP 500 matches. | Keep the clean-subprocess and lightweight-endpoint checks as regression gates. |
| `HIST-005` | `validation.tier0.s00` | Hosted CI on the original test revision failed `test_concurrent_service_initialization_keeps_ml_imports_lazy`; the test relied on ambient database state and hid child stderr. | Diagnostic run `35743413087` exposed `no such table: application_settings`; revision `3f180d2` initializes an isolated test database before service construction and passed hosted CI run `35748390899`. | No production race was established; keep the repeated clean-process regression and current CI gate. |

## Evidence and Ownership Boundaries

- Architecture documents describe structure, dependency direction, contracts,
  and persistence ownership. They remain the detailed technical authority.
- QA summaries, test files, screenshots, logs, and receipts explain how a status
  was established. They are evidence, not substitutes for current status.
  Tracked evidence is limited to compact records; bulky captures are archived
  through Git history rather than kept in the working tree.
- [`validation_campaign_ledger.md`](validation_campaign_ledger.md) is the
  subordinate slice authority for campaign order, gate scope, and remaining
  gaps.
- Implementation plans describe intended work and do not change a component's
  status until implementation and evidence exist.
- This ledger is the canonical current operational summary. When detailed
  documents disagree with a current validated artifact or source contract,
  update the affected documentation and this ledger together; do not preserve
  an obsolete failure in the active issue list.
