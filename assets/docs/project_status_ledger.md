# XREPORT Project Status Ledger

Last updated: 2026-09-25

This is the canonical high-level operational status catalog for the current
XREPORT checkout. It summarizes what is working, validated, partial, blocked,
unvalidated, or absent, and points to the detailed architecture and QA evidence
that supports each claim. It is a current-state index, not a development diary,
issue tracker replacement, release approval, or clinical-quality statement.

Validation baseline: `develop` started at
`481605b1b87035b8deb03edaefdbfc090f8f1b23`. The 2026-09-22 startup change set
was validated in that worktree before commit; see the dated Tier 0 summary.
The current validation checkout for the 2026-09-24 S31 run began at
`80e8d7aea3530c31d7557445685591533a27b3c5` (`develop`); hosted CI run
`36014118210` passed the preceding validation commit `15b45a5`. See the current
campaign ledger and S31 summary for this checkout's live model evidence.
S21/S22 application source and validation/test revision:
`ad819fdce96bfd237b1eab5586023eb68d932bab` (`develop` at that campaign).
The S23–S26 campaign began from `3846609cc68646b501cd480c2a11938fa2278f8b`.
S20 passed its focused
local API and service checks on the earlier application revision; see the [S20
summary](../QA/validation_campaign/s20/summary-20260923.md). S21 and S22 passed
on their recorded revision and promoted the dataset upload/preparation workflow
to `VALIDATED`; see the current snapshot and the dated S21/S22 summaries. Hosted
CI run [35880057345](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35880057345)
passed every configured backend and client gate on ledger revision
`a4e83cd619f9514cdf1c384a8ce47cacd16eae31`, including source/test revision
`ad819fdce96bfd237b1eab5586023eb68d932bab`. The previous CI run
`35865126628` passed every configured gate on validation/test revision
`2c18256`, including the client unit suite. The preceding docs-only revision
`caa4a41` failed run `35862434438` on two desktop-dialog assertions; the repaired
Tauri bridge fixture passes locally (`3` focused and `39` complete client tests)
and in hosted CI. S00 and S03 are now `PASS` at the recorded Tier 0 gate scope.
S02, S13, S14, and S15 were also revalidated on 2026-09-23; their dated
summaries and the validation campaign ledger retain their exact evidence scope.
On 2026-09-24, S01 and S02 were revalidated with an isolated source-launch
resource root, occupied-port refusal, controlled slow/unavailable/retry timing,
and a post-ready feature error. See the [2026-09-24 Tier 0
summary](../QA/validation_campaign/tier-0/summary-20260924.md); broader source
launcher races and packaged desktop startup remain open.
The S23–S26 changes and campaign evidence were committed as
`d665ee078ef5a878113f3db03c518834b5afdd68`; hosted CI run
[35904882475](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35904882475)
passed every configured backend and client gate. The S23 deletion follow-up
code and regression test passed locally on revision
`a00897ff74887e39a4270b933f9f3bdde8cb9b64`: source deletion now returns 409
while processed dependents exist, preserving their training samples. S24–S26
remain passed at their synthetic technical scope. See the [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md)
and [original S23–S26 campaign summary](../QA/validation_campaign/s23-s26/summary-20260923.md).
The 2026-09-24 S23 code/test revision `a00897ff74887e39a4270b933f9f3bdde8cb9b64`
initially had no hosted CI result. The subsequent validation commit
`15b45a5d276a9ebf0453b0989fb92e39e404025f` was pushed to `develop`; hosted CI
run [36014118210](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/36014118210)
passed every configured backend and client gate on that exact revision,
including PostgreSQL persistence. The full local Windows runner also passed
after a test-only S15 async-state assertion fix; see the [ISSUE-004 summary](../QA/validation_campaign/issue-004-windows-cache-20260924/summary.md).

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
  Hosted CI run `35865126628` passed every configured gate on validation/test
  revision `2c18256`, including Ruff, backend unit tests, Pyright, PostgreSQL,
  API E2E, client build/server/lint, and client unit tests. The client unit
  repair also passed locally (`3` focused and `39` complete client tests).
  S00 and S03 are `PASS` for this current Tier 0 gate scope. The prior failures
  on `caa4a41` and `5cbe2a7` are retained in the [2026-09-23 Tier 0
  summary](../QA/validation_campaign/tier-0/summary-20260923.md).
  On 2026-09-23 the focused S02 gate
  again rendered the current loading screen and ready workspace after two
  unhealthy health responses; see the [2026-09-23 Tier 0
  summary](../QA/validation_campaign/tier-0/summary-20260923.md) and the
  [validation campaign ledger](validation_campaign_ledger.md).
- S21 and S22 passed on source/test revision
  `ad819fdce96bfd237b1eab5586023eb68d932bab`. Local checks passed (10 image
  preparation unit tests, the minimum-one sampling regression, three API/UI
  E2E tests, Ruff, and `git diff --check`). A rendered partial-import preview
  showed matched/unmatched counts and persisted nothing until confirmation;
  real DistilBERT processing retained four rows at a 50% sample, then eight at
  100% under the same processed name. Both splits, missing-image rejection,
  two processing-run records, and metadata after backend restart were verified
  in isolated synthetic resources. The dataset upload/preparation workflow is
  now `VALIDATED` for this scope; no scale, training, or clinical-quality claim
  is made. See the [S21 summary](../QA/validation_campaign/s21/summary-20260923.md)
  and [S22 summary](../QA/validation_campaign/s22/summary-20260923.md).
- S23–S26 passed at their recorded scope on commit
  `d665ee078ef5a878113f3db03c518834b5afdd68`; hosted CI run
  [35904882475](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35904882475)
  passed. The 2026-09-24 S23 follow-up passed on revision
  `a00897ff74887e39a4270b933f9f3bdde8cb9b64`: source deletion now conflicts
  while a processed dependent exists and preserves its training data. S24–S26
  passed on an eight-row synthetic training path. See the [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md)
  and [S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md).
- S27 passed on 2026-09-23 against a disposable eight-row synthetic dataset.
  Full-dataset validation completed twice with all three metric families,
  returned a persisted report, and rendered the saved timestamp after a small
  completion-view fix. The focused backend suite passed (`22`), the client
  unit suite passed (`40`), and the production rebuild and in-app browser
  review passed. See the [S27 summary](../QA/validation_campaign/s27/summary-20260923.md).
- S33 passed on 2026-09-24 for the rendered one-model/one-public-image
  inference flow using an isolated exact-pinned CXRMate Multi install. Generate,
  cancel, retry, provenance, report history, edit/save/reload, copy, and export
  were observed in the in-app browser. S52's focused inference-route check
  passed its measured viewport and modal-focus scenarios after a focus-trap fix,
  but remains `PARTIAL` because reduced-motion emulation and the wider route
  matrix were not run. The current manifest comparison also found a mismatch
  in one file in the separate canonical CXRMate Multi resource; it was left
  untouched, and the provider component is now `PARTIAL` (`ISSUE-006`). S28 is
  still `UNTESTED`, S35 `FAIL / degraded`, and S37 `BLOCKED`. See the
  [S33/S52 summary](../QA/validation_campaign/s33-s52-20260924/summary.md) and
  its [gate recheck](../QA/validation_campaign/s33-s52-20260924/gate-recheck.json).
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
- S23–S26 have now been exercised in an isolated synthetic campaign. Training,
  cancellation/resume, and checkpoint registry/deletion are validated for that
  technical scope. The S23 follow-up safely blocks source deletion while a
  processed dependent exists; see the 2026-09-24 report. S27 passed for its
  small synthetic scope. S28 successful checkpoint evaluation and packaged
  desktop workflows remain validation debt.
- The earlier red runs remain in the historical ledger; the current CI result
  supersedes their S00 status.

## Current Component Ledger

### Runtime, configuration, and platform

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `runtime.source.startup` | `PARTIAL` | Windows source warm launch, current-build reuse, lightweight Node static/API proxy, rendered readiness, isolated SQLite startup, and selected port/build-freshness cases. | [2026-09-24 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260924.md); [2026-09-22 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260922.md); [2026-09-21 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260921.md); [backend smoke log](../QA/xreport-backend-smoke-20260919.log) | Permission-denied termination, post-consent PID takeover, a still-bound owner, output deletion, rebuild-after-source-change, package-lock invalidation/npm-ci recovery, and edit-during-build are not yet exercised end to end. The current managed environment denied process metadata access; the noninteractive port refusal was verified through netstat fallback, but interactive termination paths remain unvalidated. | — | 2026-09-24 | unit + integration + E2E + manual | [startup](runtime/startup.md); [system overview](architecture/system_overview.md) | Complete the remaining launcher ownership/race/failure and dependency-invalidation cases before upgrading this scoped status. |
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
| `persistence.checkpoint_registry` | `VALIDATED` | Database-owned checkpoint identity, complete-artifact registration, metadata/history listing, safe and referenced deletion, and artifact/registry cleanup. | [S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md); [checkpoint API receipt](../QA/validation_campaign/s23-s26/s26-checkpoint-api.json); [checkpoint/deletion tests](../../app/tests/e2e/test_training_api.py) | Synthetic technical checkpoint only. `ISSUE-003` was traced to incomplete test fixture history, corrected, and revalidated with no stale registrations or related warnings. | — | 2026-09-23 | unit + API E2E + manual | [persistence](architecture/persistence.md); [backend API](architecture/backend_api.md) | Revalidate registry behavior after artifact format, reference, or deletion-contract changes. |

### Domain workflows

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `workflow.dataset_upload_and_preparation` | `VALIDATED` | Synthetic non-empty upload through image matching, partial confirmation, processing, persistence, and downstream-ready processed dataset. | [S21 summary](../QA/validation_campaign/s21/summary-20260923.md); [S22 summary](../QA/validation_campaign/s22/summary-20260923.md); [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md); [original S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md); [image scanning tests](../../app/tests/unit/test_preparation_image_scanning.py); [dataset API/UI tests](../../app/tests/e2e/test_dataset_workflow_api.py); [dataset page test](../../app/tests/e2e/test_dataset_workflow_ui.py) | Evidence uses an eight-row synthetic corpus; large/real-world data, packaged desktop operation, and clinical quality remain outside this status. Source deletion returns 409 while processed dependents exist; delete those datasets first. | — | 2026-09-24 | unit + API contract + prior browser E2E + live processing + restart persistence | [workflows](operations/workflows.md); [backend API](architecture/backend_api.md) | Revalidate the deletion contract when dataset ownership changes; continue to S28 for successful checkpoint-evaluation evidence. |
| `workflow.training_and_resume` | `VALIDATED` | One-epoch real CUDA training on a tiny synthetic dataset, checkpoint creation/registration, managed cancellation after progress, worker exit, and resume with persisted history advancing by one epoch. | [S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md); [training receipt](../QA/validation_campaign/s23-s26/s24-s25-training.json); [checkpoint API evidence](../QA/validation_campaign/s23-s26/s26-checkpoint-api.json); [training API tests](../../app/tests/e2e/test_training_api.py); [training worker tests](../../app/tests/unit/test_training_stop_mechanism.py); [training memory tests](../../app/tests/unit/test_training_memory_guards.py) | Synthetic technical training does not support clinical, model-quality, or performance claims. The full dataset-to-model workflow remains bounded to this eight-row fixture and local runtime. | — | 2026-09-23 | unit + API E2E + real CUDA job + cancellation/resume + manual | [workflows](operations/workflows.md); [execution and data flow](architecture/execution_and_data_flow.md) | Revalidate on worker, checkpoint, or resume changes; continue to S27/S28 for validation and evaluation evidence. |
| `workflow.dataset_validation` | `VALIDATED` | Full-dataset validation of an eight-row synthetic dataset; text, image, and pixel metrics completed; metric values and report persisted and retrieved; completion view showed saved metadata. | [S27 summary](../QA/validation_campaign/s27/summary-20260923.md); [fixture and receipts](../QA/validation_campaign/s27/); [validation contract tests](../../app/tests/unit/test_validation_contract_metrics.py); [dataset page regression test](../../app/client/src/app/pages/dataset.page.spec.ts) | Synthetic technical evidence only. No representative-data, scale, or clinical-quality claim. The report's `artifacts` map was empty, so separate artifact-file generation was not exercised. Browser screenshot was inspected inline but not exported as a standalone file by the exposed capture API. | — | 2026-09-23 | unit + API integration + manual rendered review + production build | [workflows](operations/workflows.md); [backend API](architecture/backend_api.md) | Revalidate metric/report changes; validate separate file-artifact generation if that capability is introduced; broaden the dataset scope only with representative fixtures and appropriate acceptance criteria. |
| `workflow.checkpoint_evaluation` | `UNVALIDATED` | Compatible checkpoint evaluation, associated dataset resolution, metrics, and report persistence. | [evaluation tests](../../app/tests/unit/test_evaluation.py); [validation job tests](../../app/tests/unit/test_validation_job_semantics.py); [current S28 gate recheck](../QA/validation_campaign/s33-s52-20260924/gate-recheck.json); [2026-09-24 prerequisite probes](../QA/validation_campaign/s28/prerequisite-check-20260924.json) | Preflight and failure semantics are tested; successful live evaluation is not evidenced. Recheck found no S24 artifact or pinned BEiT snapshot in the checked canonical/campaign/cache paths. | Compatible S24 checkpoint and pinned encoder unavailable locally | 2026-09-24 | unit + prerequisite inspection | [workflows](operations/workflows.md); [backend API](architecture/backend_api.md) | Restore the pinned encoder, recreate S24, then run S28 and retrieve its persisted evaluation report. |
| `workflow.inference.generation_pipeline` | `VALIDATED` | One exact-pinned public-model install in an isolated runtime, real study generation, rendered Generate/poll/cancel/retry/review, provenance, report history, edit/save/reload, copy/export, and restart/reuse lifecycle. | [S33/S52 summary](../QA/validation_campaign/s33-s52-20260924/summary.md); [browser observations](../QA/validation_campaign/s33-s52-20260924/ui-observations.json); [S31 lifecycle summary](../QA/validation_campaign/s31/summary-20260924.md); [real generation receipt](../QA/validation_campaign/s31/real-inference-job.json); [provider tests](../../app/tests/unit/test_huggingface_provider.py) | Limited to CXRMate Multi and one public image; the canonical local model resource has a one-file manifest mismatch tracked as `ISSUE-006`. No quality, clinical, multi-provider, timeout, or persistence-failure claim. Catalogue `validation_status` remains `pending`. | — | 2026-09-24 | integration + client unit + manual browser + restart/runtime | [local inference models](runtime/local_inference_models.md); [workflows](operations/workflows.md) | Reconcile and verify the canonical snapshot through the supported pinned repair path; then cover other providers, timeout/persistence-failure behavior, and multi-case quality separately. |

### Model providers

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `inference.catalogue` | `VALIDATED` | Five pinned public entries, exact revisions, adapters, access/readiness state, and model-declared output sections, with public/custom identity separation. | [S30–S32 campaign](../QA/validation_campaign/s30-s32-20260924/summary.md); [live catalogue response](../QA/validation_campaign/s30-s32-20260924/catalog-api.json); [S31 staged/ready/deleted catalogue evidence](../QA/validation_campaign/s31/catalog-staged.json); [catalogue screenshot](../QA/validation_campaign/s30-s32-20260924/inference-catalogue-20260924.jpg); [catalogue unit tests](../../app/tests/unit/test_inference_model_catalog.py); [API E2E](../../app/tests/e2e/test_inference_api.py); [model configuration](../../app/server/configurations/inference_models.py) | S31 verified lifecycle transitions for one open public entry and retained all five public entries after isolated deletion; no custom checkpoint was registered, so live custom listing remains unobserved. | — | 2026-09-24 | unit + API E2E + live maintenance | [local inference models](runtime/local_inference_models.md); [backend API](architecture/backend_api.md) | Revalidate after manifest/catalog changes and confirm live custom listing when a compatible checkpoint is registered. |
| `inference.model.cxrmate-multi` | `PARTIAL` | Fresh isolated install of pinned CXRMate Multi TF revision, one-image Findings/Impression generation, readiness promotion, restart reuse, and delete lifecycle; canonical resource integrity rechecked. | [S33/S52 summary](../QA/validation_campaign/s33-s52-20260924/summary.md); [canonical manifest comparison](../QA/validation_campaign/s33-s52-20260924/canonical-model-recheck.json); [S31 lifecycle summary](../QA/validation_campaign/s31/summary-20260924.md); [2026-09-24 live inference](../QA/validation_campaign/s31/real-inference-job.json) | Seven of eight canonical files match the pinned manifest; `modelling_multi.py` differs in size and SHA-256. The canonical/user resource was left untouched. Fresh isolated installation and generation passed, but default canonical resource integrity is unresolved. Catalogue `validation_status` remains `pending`; no quality or clinical claim. See `ISSUE-006`. | Restore/verify the canonical file through the supported pinned repair workflow | 2026-09-24 | isolated integration + manual browser + manifest comparison | [local inference models](runtime/local_inference_models.md) | Reconcile only through the supported pinned repair workflow, verify all eight manifest hashes, and repeat canonical load/generation before strengthening status. |
| `inference.model.cxrmate-ed` | `PARTIAL` | Pinned CXRMate-ED generation with clinical context and Findings/Impression output. | [2026-09-19 receipt](../QA/inference_validation/aehrc__cxrmate-ed-68251c7605067ddbea330413aade032713fd2192.json); [failed three-case canary](../QA/inference_validation_runs/cxrmate-ed-sensitivity-20260919T154916Z.json); [current gate recheck](../QA/validation_campaign/s33-s52-20260924/gate-recheck.json) | The live catalogue reports `degraded`. The historical three cases produced only two distinct reports, and the exact approved fixture directory is absent; no replacements were used. See `ISSUE-001`. | Approved canary fixtures unavailable | 2026-09-24 | prior integration + live API + manual fixture check | [local inference models](runtime/local_inference_models.md); [commands](operations/commands_and_locations.md) | Restore the exact approved fixtures, then rerun and review the full three-case gate. |
| `inference.model.chexone` | `WORKING` | Pinned CheXOne Findings-only report generation and declared UI contract. | [2026-09-20 receipt](../QA/inference_validation/StanfordAIMI__CheXOne-0c350e6852ea08f9d9baf3b7595c1a10d4849927.json); [CheXOne contract tests](../../app/tests/e2e/test_inference_api.py) | Receipt status is passed for one fixture, but catalogue `validation_status` is `pending` and `manifest_promoted` is false. | — | 2026-09-20 | integration + manual | [local inference models](runtime/local_inference_models.md); [UI patterns](ui/components_and_patterns.md) | Rerun the current five-model aggregate and a multi-case quality gate. |
| `inference.model.cxrmate-2` | `WORKING` | Pinned CXRMate-2 Findings/Impression generation on a high-demand local runtime. | [2026-09-19 receipt](../QA/inference_validation/aehrc__cxrmate-2-aa8e2d16470e20671acf049687b4707c9bf2f2b5.json) records a passed real inference. | Catalogue `validation_status` remains `pending`; high memory/time demand is documented, not a confirmed defect. | — | 2026-09-19 | integration + manual | [local inference models](runtime/local_inference_models.md); [runtime configuration](runtime/configuration.md) | Run a multi-case quality and resource-budget check before promoting readiness. |
| `inference.model.medgemma` | `BLOCKED` | Gated MedGemma installation and local generation. | [public-model run summary](../QA/inference_validation_runs/public-inference-models-20260919T165746Z.json); [current S28/S35/S37 gate recheck](../QA/validation_campaign/s33-s52-20260924/gate-recheck.json); [live catalogue response](../QA/validation_campaign/s30-s32-20260924/catalog-api.json) | The live catalogue reports `access_policy=gated` and `not_installed`; no `HF_TOKEN` was present in the validation process or `settings/.env`. The stored credential store was not probed. No installation or generation was attempted. See `ISSUE-002`. | Authorized gated-model access and a supported credential | 2026-09-24 | live API + manual environment check | [gated access](runtime/local_inference_models.md); [configuration](runtime/configuration.md) | After access is authorized and a supported credential is available, download the exact pinned revision and capture a real inference receipt. |

### User interface and test infrastructure

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ui.settings` | `VALIDATED` | Settings route navigation, four public controls, save/reload/reset, explicit states, and responsive layout. | [settings E2E](../../app/tests/e2e/test_angular_ui.py); [2026-09-17 E2E note](../QA/e2e-validation-20260917.md); [responsive screenshots](../QA/xreport-settings-narrow-dark.png) | No current issue recorded for the exercised settings surface. | — | 2026-09-20 | E2E + manual | [UI experience](ui/experience.md); [UI patterns](ui/components_and_patterns.md) | Revalidate persistence and responsive states after route, contract, or token changes. |
| `ui.inference.catalogue_and_sections` | `VALIDATED` | Rendered inference route, model-card grid, banner behavior, responsive layout, model-declared report sections, model-dependent context controls, profile changes, and desktop catalogue/details geometry. | [S30–S32 campaign](../QA/validation_campaign/s30-s32-20260924/summary.md); [browser observations](../QA/validation_campaign/s30-s32-20260924/browser-observations.json); [catalogue screenshots](../QA/validation_campaign/s30-s32-20260924/inference-catalogue-20260924.jpg); [S33/S52 summary](../QA/validation_campaign/s33-s52-20260924/summary.md); [CheXOne UI contract test](../../app/tests/e2e/test_angular_ui.py) | The catalogue/layout scope is validated. Full generation/review UI evidence currently covers only CXRMate Multi and one public image; other provider output contracts are separate. | — | 2026-09-24 | API E2E + manual rendered review | [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Revalidate provider-declared sections against each relevant model's rendered generation flow. |
| `ui.inference.workflow` | `PARTIAL` | Rendered one-model Generate/poll/cancel/retry/review/provenance/history/edit/copy/export flow plus focused inference-route keyboard, modal, and responsive checks. | [S33/S52 summary](../QA/validation_campaign/s33-s52-20260924/summary.md); [browser observations](../QA/validation_campaign/s33-s52-20260924/ui-observations.json); [inference regression tests](../../app/client/src/app/pages/job-cancellation.pages.spec.ts); [modal focus regression test](../../app/client/src/app/components/modal-focus.directive.spec.ts); [production build](../QA/validation_campaign/s33-s52-20260924/client-production-build.log) | One provider and one public image only. The reduced-motion preference was not emulated; broader route/modal/accessibility matrix remains open. The browser download event was not surfaced, though the expected exported file was present and its contents/hash checked. The rendered screenshot was inspected inline but not saved as a standalone evidence file. | — | 2026-09-24 | client unit + production build + manual browser E2E | [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Expand coverage to other model output contracts and the remaining S52 accessibility matrix. |
| `ui.reports.history` | `VALIDATED` | Reports navigation, filtered/paginated history cards, persisted detail metadata, section-aware draft editor, explicit empty/loading/error states, and edit/reload/delete behavior. | [2026-09-22 reports validation](../QA/validation_campaign/reports-history-validation-20260922.md); [Reports route E2E](../../app/tests/e2e/test_angular_ui.py); [section helper unit tests](../../app/client/src/app/common/report-sections.spec.ts) | The CRUD browser evidence uses a deterministic disposable fixture; clinical review quality and provider-wide generation remain outside this surface. | — | 2026-09-22 | unit + E2E + manual | [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Revalidate responsive history behavior after route or editor changes. |
| `ui.startup_gate` | `VALIDATED` | Shell-level startup surface, serialized `/api/health` readiness polling, route suppression before readiness, controlled slow/unavailable/retry recovery, ready transition, ordinary post-ready feature error, responsive composition, and current built-bundle rendering. | [2026-09-24 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260924.md); [current startup screenshots](../QA/validation_campaign/tier-0/s02-20260924/); [2026-09-23 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260923.md); [startup E2E](../../app/tests/e2e/test_angular_ui.py); [frontend unit suite](../../app/client/src/app/services/startup-readiness.service.spec.ts) | Slow/unavailable timing uses a controlled browser clock and intercepted health/catalogue responses; packaged CPU/CUDA startup and real inference are not covered. | — | 2026-09-24 | unit + E2E + manual | [startup](runtime/startup.md); [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Recheck packaged CPU/CUDA startup after the next desktop validation run. |
| `ui.shell.routing_theme_guidance` | `VALIDATED` | Routed shell surfaces, root redirect, direct-load/refresh behavior, exact active navigation for parameterized child routes, Light/Dark/System theme resolution and persistence, versioned guidance dismissal/completion, and manual replay. | [2026-09-22 S10 summary](../QA/validation_campaign/s10/summary-20260922.md); [S10 E2E coverage](../../app/tests/e2e/test_angular_ui.py); [route contract](../../app/client/src/app/app.routes.ts) | The S10 browser gate used disposable report and browser-scoped validation fixtures; complete dataset/training workflows remain outside this status. | — | 2026-09-22 | E2E + manual | [UI experience](ui/experience.md); [UI patterns](ui/components_and_patterns.md) | Revalidate after route, shell, theme, or guidance contract changes. |
| `ui.dataset_and_training_surfaces` | `PARTIAL` | Dataset and Training route rendering; dataset table source-path containment and title; rendered eight-image viewer navigation; processed dataset metadata and training surface inspection; S27 successful validation summary review. | [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md); [S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md); [S27 summary](../QA/validation_campaign/s27/summary-20260923.md); [row/viewer captures](../QA/validation_campaign/s23-s26/s23-ui-observations.json); [dataset workflow E2E](../../app/tests/e2e/test_dataset_workflow_ui.py); [S10 route matrix](../QA/validation_campaign/s10/summary-20260922.md) | User-reported long Windows source path overflow was reproduced and fixed with zero-min-width ellipsis containment and full path in the title. S23's rejected-delete message and processed-first deletion order were validated at the API, but not recaptured in the browser; checkpoint evaluation UI and other complete dataset lifecycle states remain outside this scope. | — | 2026-09-23 | E2E + manual | [UI patterns](ui/components_and_patterns.md); [workflows](operations/workflows.md) | Render the blocked-delete and processed-first flow in the browser; finish the S28 evaluation view. |
| `ui.validation.report_review` | `VALIDATED` | Successful synthetic dataset report completion and review, including the persisted timestamp and metric summaries; missing-data error path remains covered by the existing E2E. | [S27 summary](../QA/validation_campaign/s27/summary-20260923.md); [persisted report receipt](../QA/validation_campaign/s27/persisted-report-response.json); [dataset page regression test](../../app/client/src/app/pages/dataset.page.spec.ts); [missing-dataset E2E](../../app/tests/e2e/test_angular_ui.py) | Eight-row synthetic technical scope only; standalone browser screenshot was not exported. Checkpoint evaluation review is not covered here. | — | 2026-09-23 | client unit + manual rendered review | [UI experience](ui/experience.md); [workflows](operations/workflows.md) | Revalidate after route/report changes; cover successful checkpoint evaluation review with S28. |
| `test.infrastructure.windows_cache` | `PARTIAL` | Repeatable Windows test startup and disposable-cache routing. | [2026-09-24 full-run summary](../QA/validation_campaign/issue-004-windows-cache-20260924/summary.md); [final runner log](../QA/validation_campaign/issue-004-windows-cache-20260924/full-run-after-s15-test-fix.log); [testing rules](coding/testing_and_quality.md) | The complete official `run_tests.bat` flow passed after fixing an S15 test-only async-state race: Python 191 passed / 3 PostgreSQL skips, frontend unit 40 passed, Angular E2E 14 passed. The runner used `runtimes/cache/pytest` and isolated basetemp without cache warnings. Git still reports access warnings for pre-existing protected cache paths outside that root; they were not cleaned. | Ownership/access and safe disposition of historical protected cache paths remain unverified; the configured runner is validated. | 2026-09-24 | full Windows runner + API E2E + browser E2E | [startup](runtime/startup.md); [commands](operations/commands_and_locations.md) | Resolve access/ownership for the historical paths before attempting cleanup; keep using the validated official runner and configured runtime caches. |

## Open Issues

Severity is deliberately separate from functional status: a `PARTIAL` component
may have a `LOW` issue, and a `BLOCKED` component may have no software defect.
Only actionable current problems belong here.

| ID | Affected Component | Severity | Concise Description | Current Impact | Reproduction or Evidence | Suspected Cause | Blocker | Remediation Status | Required Revalidation | Related Documentation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ISSUE-001` | `inference.model.cxrmate-ed` | `HIGH` | The real three-case sensitivity canary completed but produced only two distinct report texts, so the catalogue remains `degraded`. | CXRMate-ED remains selectable only as an unverified research draft; quality promotion is not justified. | [historical canary](../QA/inference_validation_runs/cxrmate-ed-sensitivity-20260919T154916Z.json): `completed_cases=3`, `reports_all_distinct=false`, `unique_report_count=2`; [current recheck](../QA/validation_campaign/s33-s52-20260924/gate-recheck.json) confirms degraded status and missing exact fixture directory. | Not established by the evidence. | Approved exact canary fixtures are unavailable. | `OPEN` — preserve the degraded warning and do not promote the manifest. | Restore the approved fixtures, remediate or accept the canary finding, then rerun and review all outputs against the quality gate. | [local inference models](runtime/local_inference_models.md); [canary command](operations/commands_and_locations.md) |
| `ISSUE-002` | `inference.model.medgemma` | `MEDIUM` | The gated MedGemma entry cannot be installed or generated without authorized provider access and a supported credential. | One of the five catalogue entries is unavailable in the current environment. | [current recheck](../QA/validation_campaign/s33-s52-20260924/gate-recheck.json): `access_policy=gated`, `not_installed`, no `HF_TOKEN` in process/settings; stored credential store not probed; no install/generation attempted. | Not a confirmed software defect. | Authorized gated access and a supported credential. | `BLOCKED` — waiting for authorized access configuration. | Download the exact pinned revision, run real inference, and capture a receipt before changing status. | [gated access](runtime/local_inference_models.md); [configuration](runtime/configuration.md) |
| `ISSUE-006` | `inference.model.cxrmate-multi` | `MEDIUM` | The canonical CXRMate Multi TF resource does not fully match its pinned S31 snapshot manifest: `modelling_multi.py` is 18,493 bytes / SHA-256 `4b06e742bbd3f27b6fd61ef0adfcc693ab7b944fc25b41cc5c6cddc45459f5a9`, expected 17,211 bytes / SHA-256 `528e8d471658802c0ba40fd37a8839cfcd29c342179adeb412f22e04bb7856ee`. | Seven of eight files match. The isolated exact-pinned install and S33 generation passed, but canonical resource integrity and behavior are not established. | [canonical model comparison](../QA/validation_campaign/s33-s52-20260924/canonical-model-recheck.json); [S33/S52 observations](../QA/validation_campaign/s33-s52-20260924/ui-observations.json) | Origin of the differing canonical file is unknown. | Canonical user resource was preserved without modification; use the supported pinned repair workflow before claiming canonical integrity. | `OPEN` — no canonical model file was changed during this validation. | Repair from the exact pinned source, recheck all eight hashes, then repeat canonical load and inference. | [local inference models](runtime/local_inference_models.md) |
| `ISSUE-004` | `test.infrastructure.windows_cache` | `LOW` | Legacy unconfigured pytest runs warned on protected cache paths. The full official `run_tests.bat` now passes through configured runtime caches without pytest cache warnings. | Test execution is validated; Git directory scans still warn when they encounter some pre-existing protected cache paths outside the configured root. | [2026-09-24 runner summary and logs](../QA/validation_campaign/issue-004-windows-cache-20260924/summary.md); [S30–S32 campaign](../QA/validation_campaign/s30-s32-20260924/summary.md); earlier [settings QA](../QA/settings-migration-validation-20260917.md), [E2E QA](../QA/e2e-validation-20260917.md), and [S12 summary](../QA/validation_campaign/s12/summary-20260922.md). | Historical cache directory ACL/ownership, not the official runner configuration. | Filesystem permission/ownership of legacy cache paths. | `OPEN` — official runner is validated; legacy protected paths remain inaccessible and were left unchanged. | Verify ownership/access and safe disposition of the historical paths before cleanup; do not remove cache residue without confirmed ownership. | [testing rules](coding/testing_and_quality.md); [startup](runtime/startup.md) |

## Resolved Issues

| ID | Affected Component | Resolution and Revalidation | Evidence |
| --- | --- | --- | --- |
| `ISSUE-003` | `persistence.checkpoint_registry` | The checkpoint E2E fixture wrote an empty session history, causing complete-artifact validation to warn and making API cleanup ineffective; its fixture now contains one epoch of loss/validation history, and cleanup always reconciles registry and artifact state. Isolated checkpoint listing after the suite was empty with zero matching stale registrations or warnings. | [S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md); [root-cause receipt](../QA/validation_campaign/s23-s26/issue-003.json); [training API tests](../../app/tests/e2e/test_training_api.py) |
| `ISSUE-005` | `workflow.dataset_upload_and_preparation` | Source deletion now checks dependent processing runs within the deletion transaction and returns HTTP 409 with dependent dataset names. The source, records, and training samples remain usable until the processed dependents are deleted; processed-then-source deletion finishes with clean foreign-key and SQLite integrity checks. | [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md); [deletion regression test](../../app/tests/unit/test_dataset_deletion.py) |

## Validation Debt

Validation debt is not a defect list. It identifies important behavior whose
current confidence is too low or too narrow to support a stronger status.

| Component | Current Confidence | Missing Validation | Priority |
| --- | --- | --- | --- |
| `runtime.desktop.packaged` | Low | CPU/CUDA portable and MSI build, launch, readiness, shutdown, data-root isolation, and artifact verification. | High |
| `workflow.checkpoint_evaluation` | Low | Successful compatible checkpoint evaluation and report retrieval. | Medium |
| `workflow.inference.generation_pipeline` | Medium | Other-provider browser flows, timeout and persistence-failure behavior, broader reuse conditions, and multi-case quality. | High |
| `inference.model.*` quality | Low | Multi-case quality gates for all public models; CXRMate-ED currently has a failed gate. | High |
| `ui.inference.workflow` | Medium | Other-provider output contracts, reduced-motion runtime behavior, broader route/modal accessibility matrix, and persistent screenshot artifacts for rendered acceptance. | High |
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
