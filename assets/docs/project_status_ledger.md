# XREPORT Project Status Ledger

Last updated: 2026-09-28

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
launcher races remain open. The 2026-09-25 CPU packaged desktop build and smoke
now pass at technical scope, while CUDA packaging and native WebView interaction
remain open.
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
  were observed in the in-app browser. S52's initial focused inference-route
  check passed its measured viewport and modal-focus scenarios after a focus-trap
  fix; the later 2026-09-28 matrix supersedes its then-open reduced-motion and
  wider-route limitations. The current manifest comparison also found a mismatch
  in one file in the separate canonical CXRMate Multi resource; it was left
  untouched, and the provider component is now `PARTIAL` (`ISSUE-006`). On
  2026-09-25, S28 and S34 passed at synthetic technical scope, S35 remained
  `FAIL / degraded` after its exact-fixture rerun, and S37 remained `BLOCKED`.
  See the
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
  small synthetic scope, and S28/S34 now pass for their eight-row synthetic
  checkpoint-evaluation and custom-inference workflows. The CPU packaged
  desktop workflow now passes its technical artifact/smoke scope; CUDA is now
  technically exercised but native WebView interaction and no-GPU fallback
  remain validation debt.
- The earlier red runs remain in the historical ledger; the current CI result
  supersedes their S00 status.

- On 2026-09-27, the official CUDA desktop release path built and verified portable/MSI artifacts from source commit `383cba0b85ad374a1fa8581b2ca3d2eee045932b`. The packaged CUDA backend passed startup/readiness/health/frontend smoke and completed one real CXRMate Multi inference on the RTX 3060; job and persisted-history provenance both recorded `resolved_device=cuda:0`, `cuda_available=true`, and `cuda_used=true`. The canonical model resource remains untouched under `ISSUE-006`; the exact pinned remote-code file was used only in the disposable probe snapshot. S42 is `PARTIAL` because no-GPU fallback and native WebView navigation remain untested. See the [S42 summary](../QA/validation_campaign/s42-cuda-20260927/summary-20260927.md) and [packaged receipt](../QA/validation_campaign/s42-cuda-20260927/packaged-cuda-inference-receipt.json).

- On 2026-09-28, the current launcher and job services were revalidated as a bounded S43/S51 slice. The official launcher started real FastAPI/Node listeners that returned HTTP 200, and `KillProcesses` stopped their service-owned trees through `/T /F`, leaving no checked PIDs or 5003/8003 listeners. The focused job regression produced one accepted and one typed conflict for each same-type duplicate, then observed every distinct job-type pair overlap and complete without stale jobs. S43 and S51 remain `PARTIAL`: packaged/Tauri ownership, canonical release cleanup, a broader real heavy-worker resource matrix, UI close/reopen, per-job attribution, and repeated no-stuck heavy-worker coverage remain open. See the [S43/S51 summary](../QA/validation_campaign/s43-s51-20260928/summary-20260928.md) and [live listener receipt](../QA/validation_campaign/s43-s51-20260928/s43-live-listener-receipt.json).

- The 2026-09-28 S50/S52 recovery and UI-resilience slice supersedes the earlier partial recheck for its covered boundaries. S50 is now `PASS` for the current source launcher with disposable SQLite: setting/report restart persistence, deterministic schema-drift fail-closed startup, sanitized unavailable/retry rendering, no silent database recreation, backup restoration, and post-restore integrity/readback all passed. S52 remains `PARTIAL` only because spoken Narrator/Speech Recap output was not observed; its 25 route/viewport cases passed responsive layout, scroll, modal bounds, focus containment, Escape restoration, reduced-motion behavior, screenshots, and clean browser collectors. The adjacent S51 follow-up remains partial after 22 focused tests because broader real heavy-worker/resource contention is only partially measured. See the [S50/S52 summary](../QA/validation_campaign/s50-s52-20260928/summary-20260928.md), [structured receipt](../QA/validation_campaign/s50-s52-20260928/s50-s52-validation-receipt.json), [fail-closed evidence](../QA/validation_campaign/s50-s52-20260928/s50-startup-fail-closed.log), and [committed E2E coverage](../../app/tests/e2e/test_angular_ui.py). Protected historical cache ACL paths remain unchanged under `ISSUE-004`; packaged/native desktop, model-quality, provider-access, and representative-performance boundaries remain as recorded below.

- On HEAD `99dfa44bc1560d1fb8e3e468e68ddf6e25ba1a7b`, the root-level resource move was rechecked through the official source launcher with a process-level disposable resource override; SQLite initialization, backend health, frontend serving, and the frontend health proxy returned `200`. The current local S03 recheck passed `166` backend unit tests, Ruff, Pyright, Angular build/lint, and `43` client unit tests. A bounded live S51/S53 probe then ran validation, checkpoint evaluation, and one-epoch CUDA training concurrently against the disposable eight-row S28 resource; all jobs completed, evaluation/training overlapped for about `18.024s`, GPU sampling peaked at `3,906/6,144 MiB` and `100%`, and no running jobs remained after cleanup. Dataset processing took `1.085s` for `8` samples. See the [current-head S51/S53 summary](../QA/validation_campaign/s51-s53-20260928/summary-20260928.md) and [runtime contention receipt](../QA/validation_campaign/s51-s53-20260928/current-head-runtime-contention-receipt.json). S51/S53 remain `PARTIAL` for the documented synthetic, single-run, no-UI/no-packaged/no-GPU scope.

- The 2026-09-28 S23 browser slice started from HEAD `136ec70f51f73f533485bd6a999177307cb28035`. The official source launcher started the current backend/frontend stack against a disposable SQLite root. The in-app browser rendered the Dataset and Training routes with an eight-row source/processed fixture. The focused S23 browser regression rendered the dependent-name HTTP 409 conflict, deleted the processed dataset first, and then deleted the source; the existing S23 viewer/long-path test and the backend deletion regression also passed. See the [S23 browser deletion summary](../QA/validation_campaign/s23-delete-ui-20260928/summary-20260928.md). The disposable root was removed and ports `5003`/`8003` were clear after cleanup.

## Current Component Ledger

### Runtime, configuration, and platform

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `runtime.source.startup` | `PARTIAL` | Windows source warm launch, current-build reuse, lightweight Node static/API proxy, rendered readiness, isolated SQLite startup, deterministic incompatible-schema fail-closed startup, selected port/build-freshness cases, and current root-level resource initialization. | [current-head S51/S53 summary](../QA/validation_campaign/s51-s53-20260928/summary-20260928.md); [current-head runtime receipt](../QA/validation_campaign/s51-s53-20260928/current-head-runtime-contention-receipt.json); [2026-09-28 S50/S52 summary](../QA/validation_campaign/s50-s52-20260928/summary-20260928.md); [fail-closed evidence](../QA/validation_campaign/s50-s52-20260928/s50-startup-fail-closed.log); [2026-09-24 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260924.md); [2026-09-22 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260922.md); [2026-09-21 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260921.md); [backend smoke log](../QA/xreport-backend-smoke-20260919.log) | Permission-denied termination, post-consent PID takeover, a still-bound owner, output deletion, rebuild-after-source-change, package-lock invalidation/npm-ci recovery, and edit-during-build are not yet exercised end to end. The current source failure path was validated for deterministic schema drift and listener cleanup, but interactive ownership/race paths remain unvalidated. | — | 2026-09-28 | unit + integration + E2E + manual | [startup](runtime/startup.md); [system overview](architecture/system_overview.md) | Complete the remaining launcher ownership/race/failure and dependency-invalidation cases before upgrading this scoped status. |
| `runtime.desktop.packaged` | `PARTIAL` | Tauri CPU/CUDA runtime extraction, portable/MSI packaging, startup, shutdown, user-data-root isolation, and one packaged CUDA inference with GPU provenance. | [S40/S41 summary](../QA/validation_campaign/s40-s41-desktop-20260925/summary.md); [S42 summary](../QA/validation_campaign/s42-cuda-20260927/summary-20260927.md); [CPU runtime audit](../QA/desktop/runtime-cpu-3.1.0.json); [CUDA runtime audit](../QA/desktop/runtime-cuda-3.1.0.json); [CPU/CUDA artifact verification](../QA/desktop/verification-cpu-3.1.0.json); [CUDA artifact verification](../QA/desktop/verification-cuda-3.1.0.json); [CPU packaged smoke](../QA/desktop/smoke-cpu-3.1.0.json); [CUDA packaged smoke](../QA/desktop/smoke-cuda-3.1.0.json); [CUDA inference receipt](../QA/validation_campaign/s42-cuda-20260927/packaged-cuda-inference-receipt.json); [desktop packaging tests](../../app/tests/unit/test_desktop_packaging.py) | CPU and CUDA portable/MSI artifacts, packaged startup/health/frontend serving, cleanup, and CUDA execution provenance passed for their recorded technical scopes. Native WebView navigation, no-GPU fallback, representative performance, and cross-variant benchmark equivalence remain open; ISSUE-006 keeps canonical CXRMate Multi integrity unresolved. | No-GPU lane and a targetable native WebView validation surface; supported pinned repair workflow for the canonical model | 2026-09-27 | unit + build + artifact verification + packaged API smoke + packaged real inference + manual boundary probe | [deployment](runtime/deployment.md); [runtime modes](runtime/modes.md) | Exercise no-GPU fallback and native WebView coverage, then repair/reverify the canonical model resource before strengthening this status. |
| `runtime.containerized` | `NOT_IMPLEMENTED` | Containerized runtime or image-based deployment. | [runtime modes](runtime/modes.md) explicitly records this mode as absent. | No container build, image, or deployment contract exists. | — | — | None | [runtime modes](runtime/modes.md); [deployment](runtime/deployment.md) | Define a supported container contract only if deployment scope expands. |
| `configuration.runtime_settings` | `VALIDATED` | Database-backed public settings: seed, filesystem access, polling interval, and inference timeout, including save, reload, reset, and allowlisting. | [settings validation](../QA/settings-migration-validation-20260917.md); [settings E2E](../../app/tests/e2e/test_settings_api.py); [settings page E2E](../../app/tests/e2e/test_angular_ui.py) | Hidden infrastructure, credential, and static model-policy values remain intentionally unavailable to the Settings API. | — | 2026-09-17 | unit + integration + E2E + manual | [configuration](runtime/configuration.md); [persistence](architecture/persistence.md); [UI experience](ui/experience.md) | Revalidate save/reset and migration behavior after settings or schema changes. |
| `security.authentication` | `NOT_IMPLEMENTED` | API authentication and authorization. | [execution and data flow](architecture/execution_and_data_flow.md) states that no auth layer is implemented. | Current security boundary is trusted local use; external exposure is not covered. | — | — | None | [execution and data flow](architecture/execution_and_data_flow.md); [runtime modes](runtime/modes.md) | Define and validate a threat model before supporting non-local deployment. |

### Backend, jobs, and persistence

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `backend.api.contracts` | `VALIDATED` | Tested health, model catalogue, dataset status/name, checkpoint, settings, and inference-history list/detail/filter/error surfaces plus typed error behavior. | [2026-09-22 reports validation](../QA/validation_campaign/reports-history-validation-20260922.md); [backend API tests](../../app/tests/e2e/test_inference_api.py); [OpenAPI tests](../../app/tests/unit/test_openapi_schema.py) | This status covers the exercised surface, not every endpoint or every long-running happy path. | — | 2026-09-22 | unit + integration + E2E | [backend API](architecture/backend_api.md); [system overview](architecture/system_overview.md) | Extend endpoint-level E2E coverage when a route or response contract changes. |
| `backend.jobs.lifecycle` | `VALIDATED` | Generic job start, list/filter, poll, completion, cancellation, unknown-job responses, typed terminal failure, and persistence-failure semantics. | [2026-09-22 S12 summary](../QA/validation_campaign/s12/summary-20260922.md); [API lifecycle tests](../../app/tests/integration/test_job_lifecycle_api.py); [job failure tests](../../app/tests/unit/test_job_failure_semantics.py); [job cancellation tests](../../app/tests/unit/test_job_cancellation_semantics.py) | S12 uses deterministic in-process runners; feature-specific heavy workflows remain outside this component scope. | — | 2026-09-22 | unit + integration | [execution and data flow](architecture/execution_and_data_flow.md); [backend API](architecture/backend_api.md) | Revalidate the affected job path after changes to a feature service or polling contract. |
| `backend.ml_import_boundaries` | `VALIDATED` | Lightweight service imports and concurrent service construction keep the listed Keras/PyTorch/Transformers/provider modules unloaded across ten fresh subprocess runs with an initialized isolated database. | [2026-09-22 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260922.md); [import-boundary tests](../../app/tests/unit/test_ml_import_boundaries.py); hosted CI run `35748390899` | The assertion covers the named import boundary and service factories, not every endpoint or later ML job. | — | 2026-09-22 | unit + hosted CI | [execution and data flow](architecture/execution_and_data_flow.md); [troubleshooting](operations/troubleshooting.md) | Revalidate after changes to service initialization or optional-ML import boundaries. |
| `persistence.sqlite_migrations` | `VALIDATED` | SQLite startup/initialization reaches the checked-in Alembic head `e91a4f6c2d73`, persists application settings and an inference report across restart, fails closed for deterministic incompatible schema drift, and supports backup restoration with clean integrity. | [2026-09-28 S50/S52 summary](../QA/validation_campaign/s50-s52-20260928/summary-20260928.md); [structured receipt](../QA/validation_campaign/s50-s52-20260928/s50-s52-validation-receipt.json); [2026-09-23 S13 summary](../QA/validation_campaign/s13/summary-20260923.md); [database initialization tests](../../app/tests/unit/test_database_initialization.py); [2026-09-22 reports validation](../QA/validation_campaign/reports-history-validation-20260922.md) | Existing unversioned or incompatible schemas intentionally fail closed. Crash recovery, partial migration interruption, filesystem corruption, and PostgreSQL recovery remain outside S50. | — | 2026-09-28 | unit + integration + manual rendered browser | [persistence](architecture/persistence.md); [architecture review](architecture/architecture_review.md) | Recheck migration upgrade and rollback safety for every new revision. |
| `persistence.inference_history` | `VALIDATED` | Durable session list/detail, generated-versus-edited report text, atomic updates, restart persistence of a synthetic run with linked report, and cascade deletion using the existing `InferenceRun` and `InferenceReport` entities. | [2026-09-23 S13 summary](../QA/validation_campaign/s13/summary-20260923.md); [2026-09-22 reports validation](../QA/validation_campaign/reports-history-validation-20260922.md); [repository persistence tests](../../app/tests/unit/test_repository_persistence.py); [backend API tests](../../app/tests/e2e/test_inference_api.py) | No current issue recorded for the scoped CRUD contract. | — | 2026-09-23 | unit + integration + E2E + manual | [persistence](architecture/persistence.md); [backend API](architecture/backend_api.md) | Revalidate the history contract after schema or API changes. |
| `persistence.postgresql` | `VALIDATED` | PostgreSQL 16 application migrations reach Alembic head; schema contract, settings persistence across reinitialization, concurrent initialization, advisory locks, and sanitized connection failure pass. | [2026-09-23 S14 summary](../QA/validation_campaign/s14/summary-20260923.md); [PostgreSQL contract tests](../../app/tests/integration/test_persistence_contract.py); [CI run 35841152401](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35841152401) | Scope is the exercised PostgreSQL 16 contract; it does not establish behavior for every production database configuration. | — | 2026-09-23 | unit + integration + hosted CI | [persistence](architecture/persistence.md); [deployment](runtime/deployment.md) | Revalidate after migration, connection, or database-initialization changes. |
| `persistence.checkpoint_registry` | `VALIDATED` | Database-owned checkpoint identity, complete-artifact registration, metadata/history listing, safe and referenced deletion, and artifact/registry cleanup. | [S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md); [checkpoint API receipt](../QA/validation_campaign/s23-s26/s26-checkpoint-api.json); [checkpoint/deletion tests](../../app/tests/e2e/test_training_api.py) | Synthetic technical checkpoint only. `ISSUE-003` was traced to incomplete test fixture history, corrected, and revalidated with no stale registrations or related warnings. | — | 2026-09-23 | unit + API E2E + manual | [persistence](architecture/persistence.md); [backend API](architecture/backend_api.md) | Revalidate registry behavior after artifact format, reference, or deletion-contract changes. |

### Domain workflows

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `workflow.dataset_upload_and_preparation` | `VALIDATED` | Synthetic non-empty upload through image matching, partial confirmation, processing, persistence, and downstream-ready processed dataset. | [S21 summary](../QA/validation_campaign/s21/summary-20260923.md); [S22 summary](../QA/validation_campaign/s22/summary-20260923.md); [S23 browser deletion summary](../QA/validation_campaign/s23-delete-ui-20260928/summary-20260928.md); [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md); [original S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md); [image scanning tests](../../app/tests/unit/test_preparation_image_scanning.py); [dataset API/UI tests](../../app/tests/e2e/test_dataset_workflow_api.py); [dataset page test](../../app/tests/e2e/test_dataset_workflow_ui.py) | Evidence uses an eight-row synthetic corpus; large/real-world data, packaged desktop operation, and clinical quality remain outside this status. Current browser evidence confirms that source deletion returns 409 while processed dependents exist and that processed-first deletion completes. | — | 2026-09-28 | unit + API contract + browser E2E + live processing + restart persistence | [workflows](operations/workflows.md); [backend API](architecture/backend_api.md) | Broaden only with representative fixtures and an explicit scale/quality scope after the passed S28 evaluation. |
| `workflow.training_and_resume` | `VALIDATED` | One-epoch real CUDA training on a tiny synthetic dataset, checkpoint creation/registration, managed cancellation after progress, worker exit, and resume with persisted history advancing by one epoch; same-type concurrent admission rejects the second start atomically across training, dataset processing, validation, and checkpoint evaluation; all six distinct job-type pairs now have synthetic overlap/completion evidence; the current-head live probe also completed concurrent evaluation/training without a stale active job. | [S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md); [training receipt](../QA/validation_campaign/s23-s26/s24-s25-training.json); [checkpoint API evidence](../QA/validation_campaign/s23-s26/s26-checkpoint-api.json); [current-head S51/S53 summary](../QA/validation_campaign/s51-s53-20260928/summary-20260928.md); [current-head runtime receipt](../QA/validation_campaign/s51-s53-20260928/current-head-runtime-contention-receipt.json); [training API tests](../../app/tests/e2e/test_training_api.py); [training worker tests](../../app/tests/unit/test_training_stop_mechanism.py); [training memory tests](../../app/tests/unit/test_training_memory_guards.py); [S43/S51 summary](../QA/validation_campaign/s43-s51-20260928/summary-20260928.md); [concurrency regression](../../app/tests/unit/test_job_start_concurrency.py) | Synthetic technical training does not support clinical, model-quality, or performance claims. The full dataset-to-model workflow remains bounded to this eight-row fixture and local runtime; UI close/reopen, broader real resource contention, per-job attribution, and repeated no-stuck heavy-worker behavior remain open under S51. | — | 2026-09-28 | unit + API E2E + real CUDA job + cancellation/resume + manual + concurrency regression | [workflows](operations/workflows.md); [execution and data flow](architecture/execution_and_data_flow.md) | Revalidate on worker, checkpoint, or resume changes; repeat the remaining real heavy-worker/resource-contention matrix under S51. |
| `workflow.dataset_validation` | `VALIDATED` | Full-dataset validation of an eight-row synthetic dataset; text, image, and pixel metrics completed; metric values and report persisted and retrieved; completion view showed saved metadata; current-head rerun returned the expected `104` total words, `20` unique words, and `64x64` images. | [S27 summary](../QA/validation_campaign/s27/summary-20260923.md); [current-head runtime receipt](../QA/validation_campaign/s51-s53-20260928/current-head-runtime-contention-receipt.json); [fixture and receipts](../QA/validation_campaign/s27/); [validation contract tests](../../app/tests/unit/test_validation_contract_metrics.py); [dataset page regression test](../../app/client/src/app/pages/dataset.page.spec.ts) | Synthetic technical evidence only. No representative-data, scale, or clinical-quality claim. The report's `artifacts` map was empty, so separate artifact-file generation was not exercised. Browser screenshot was inspected inline but not exported as a standalone file by the exposed capture API. | — | 2026-09-28 | unit + API integration + manual rendered review + production build | [workflows](operations/workflows.md); [backend API](architecture/backend_api.md) | Revalidate metric/report changes; validate separate file-artifact generation if that capability is introduced; broaden the dataset scope only with representative fixtures and appropriate acceptance criteria. |
| `workflow.checkpoint_evaluation` | `VALIDATED` | Compatible checkpoint evaluation, associated dataset resolution, metrics, and report persistence on an eight-row synthetic CUDA workflow; current-head rerun retrieved the persisted report with `loss=2.0254554748535156` and `accuracy=0.8571428656578064`. | [S28/S34/S53 summary](../QA/validation_campaign/s28-release-20260925/summary.md); [current-head runtime receipt](../QA/validation_campaign/s51-s53-20260928/current-head-runtime-contention-receipt.json); [workflow receipts](../QA/validation_campaign/s28-release-20260925/workflow-api-receipts.json); [encoder receipt](../QA/validation_campaign/s28/encoder-download-20260925.json); [evaluation tests](../../app/tests/unit/test_evaluation.py); [validation job tests](../../app/tests/unit/test_validation_job_semantics.py) | The current technical path passed, but the evidence is synthetic and one epoch only; no representative-data, clinical-quality, scale, separate artifact-file, or packaged-release claim is made. | — | 2026-09-28 | unit + live CUDA job + API + manual rendered review | [workflows](operations/workflows.md); [backend API](architecture/backend_api.md) | Broaden only with representative fixtures and an explicit quality/performance scope. |
| `workflow.inference.generation_pipeline` | `VALIDATED` | One exact-pinned public-model install in an isolated runtime, real study generation, rendered Generate/poll/cancel/retry/review, provenance, report history, edit/save/reload, copy/export, and restart/reuse lifecycle. | [S33/S52 summary](../QA/validation_campaign/s33-s52-20260924/summary.md); [browser observations](../QA/validation_campaign/s33-s52-20260924/ui-observations.json); [S31 lifecycle summary](../QA/validation_campaign/s31/summary-20260924.md); [real generation receipt](../QA/validation_campaign/s31/real-inference-job.json); [provider tests](../../app/tests/unit/test_huggingface_provider.py) | Limited to CXRMate Multi and one public image; the canonical local model resource has a one-file manifest mismatch tracked as `ISSUE-006`. No quality, clinical, multi-provider, timeout, or persistence-failure claim. Catalogue `validation_status` remains `pending`. | — | 2026-09-24 | integration + client unit + manual browser + restart/runtime | [local inference models](runtime/local_inference_models.md); [workflows](operations/workflows.md) | Reconcile and verify the canonical snapshot through the supported pinned repair path; then cover other providers, timeout/persistence-failure behavior, and multi-case quality separately. |

### Model providers

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `inference.catalogue` | `VALIDATED` | Five pinned public entries, exact revisions, adapters, access/readiness state, and model-declared output sections, with public/custom identity separation. | [S30–S32 campaign](../QA/validation_campaign/s30-s32-20260924/summary.md); [live catalogue response](../QA/validation_campaign/s30-s32-20260924/catalog-api.json); [S31 staged/ready/deleted catalogue evidence](../QA/validation_campaign/s31/catalog-staged.json); [catalogue screenshot](../QA/validation_campaign/s30-s32-20260924/inference-catalogue-20260924.jpg); [catalogue unit tests](../../app/tests/unit/test_inference_model_catalog.py); [API E2E](../../app/tests/e2e/test_inference_api.py); [model configuration](../../app/server/configurations/inference_models.py) | S31 verified lifecycle transitions for one open public entry and retained all five public entries after isolated deletion; no custom checkpoint was registered, so live custom listing remains unobserved. | — | 2026-09-24 | unit + API E2E + live maintenance | [local inference models](runtime/local_inference_models.md); [backend API](architecture/backend_api.md) | Revalidate after manifest/catalog changes and confirm live custom listing when a compatible checkpoint is registered. |
| `inference.model.cxrmate-multi` | `PARTIAL` | Fresh isolated install of pinned CXRMate Multi TF revision, one-image Findings/Impression generation, readiness promotion, restart reuse, delete lifecycle, and one packaged CUDA inference with persisted device provenance; canonical resource integrity rechecked. | [S42 packaged inference receipt](../QA/validation_campaign/s42-cuda-20260927/packaged-cuda-inference-receipt.json); [S42 summary](../QA/validation_campaign/s42-cuda-20260927/summary-20260927.md); [S33/S52 summary](../QA/validation_campaign/s33-s52-20260924/summary.md); [canonical manifest comparison](../QA/validation_campaign/s33-s52-20260924/canonical-model-recheck.json); [S31 lifecycle summary](../QA/validation_campaign/s31/summary-20260924.md); [2026-09-24 live inference](../QA/validation_campaign/s31/real-inference-job.json) | Seven of eight canonical files match the pinned manifest; `modelling_multi.py` differs in size and SHA-256. The canonical/user resource was left untouched. The disposable exact-pinned snapshot and packaged CUDA generation passed, but default canonical resource integrity is unresolved. Catalogue `validation_status` remains `pending`; no quality or clinical claim. See `ISSUE-006`. | Restore/verify the canonical file through the supported pinned repair workflow | 2026-09-27 | isolated integration + packaged real inference + manual browser + manifest comparison | [local inference models](runtime/local_inference_models.md) | Reconcile only through the supported pinned repair workflow, verify all eight manifest hashes, and repeat canonical load/generation before strengthening status. |
| `inference.model.cxrmate-ed` | `PARTIAL` | Pinned CXRMate-ED generation with clinical context and Findings/Impression output, including the exact three-case sensitivity rerun. | [2026-09-26 S35 receipt](../QA/validation_campaign/s35-s36-20260926/s35-cxrmate-ed-technical-receipt.json); [case manifest](../QA/validation_campaign/s35-s36-20260926/case-manifest.json); [manual review](../QA/validation_campaign/s35-s36-20260926/manual-review.json) | The exact pinned run completed all three cases with valid, distinct input bytes/dimensions/provenance but produced only two distinct full reports, so the live catalogue remains `degraded` and Generate remains disabled. This is classified as model behavior; `ISSUE-001` remains open and no quality promotion is justified. | — | 2026-09-26 | isolated integration + live API + manual browser + canary | [local inference models](runtime/local_inference_models.md); [commands](operations/commands_and_locations.md) | Investigate duplicate outputs across distinct fixtures, then rerun and review all outputs against this exact quality gate before promotion. |
| `inference.model.chexone` | `PASS` | Pinned CheXOne Findings-only report generation and declared UI contract. | [2026-09-26 aggregate receipt](../QA/validation_campaign/s35-s36-20260926/s36a-chexone-aggregate-technical-receipt.json); [manual review](../QA/validation_campaign/s35-s36-20260926/manual-review.json); [rendered catalogue observation](../QA/validation_campaign/s35-s36-20260926/browser-catalogue-observation.json); [historical one-case receipt](../QA/validation_campaign/s35-s36-20260926/historical-chexone-single-case-receipt.json); [CheXOne contract tests](../../app/tests/e2e/test_inference_api.py) | Exact revision `0c350e6852ea08f9d9baf3b7595c1a10d4849927` passed the independent three-case technical run, resource/reuse sentinel, conservative review, and current rendered catalogue check. Catalogue `validation_status=passed`; output remains research-only Findings drafting with no diagnostic-accuracy or release-readiness claim. | — | 2026-09-26 | isolated integration + manual review + browser catalogue | [local inference models](runtime/local_inference_models.md); [UI patterns](ui/components_and_patterns.md) | Extend with labelled-reference and broader representative quality studies before any clinical-performance claim. |
| `inference.model.cxrmate-2` | `PASS` | Pinned CXRMate-2 Findings/Impression generation on a high-demand local runtime. | [2026-09-26 aggregate receipt](../QA/validation_campaign/s35-s36-20260926/s36b-cxrmate2-aggregate-technical-receipt.json); [manual review](../QA/validation_campaign/s35-s36-20260926/manual-review.json); [rendered catalogue observation](../QA/validation_campaign/s35-s36-20260926/browser-catalogue-observation.json); [historical one-case receipt](../QA/validation_campaign/s35-s36-20260926/historical-cxrmate2-single-case-receipt.json) | Exact revision `aa8e2d16470e20671acf049687b4707c9bf2f2b5` passed the independent three-case technical run, very-high resource/reuse sentinel, conservative review, and current rendered catalogue check. Catalogue `validation_status=passed`; output remains research-only Findings/Impression drafting with no diagnostic-accuracy or release-readiness claim. | — | 2026-09-26 | isolated integration + manual review + browser catalogue | [local inference models](runtime/local_inference_models.md); [runtime configuration](runtime/configuration.md) | Extend with labelled-reference and broader representative quality studies before any clinical-performance claim. |
| `inference.model.medgemma` | `BLOCKED` | Gated MedGemma installation and local generation. | [public-model run summary](../QA/inference_validation_runs/public-inference-models-20260919T165746Z.json); [current S28/S35/S37 gate recheck](../QA/validation_campaign/s33-s52-20260924/gate-recheck.json); [live catalogue response](../QA/validation_campaign/s30-s32-20260924/catalog-api.json) | The live catalogue reports `access_policy=gated` and `not_installed`; no `HF_TOKEN` was present in the validation process or `settings/.env`. The stored credential store was not probed. No installation or generation was attempted. See `ISSUE-002`. | Authorized gated-model access and a supported credential | 2026-09-24 | live API + manual environment check | [gated access](runtime/local_inference_models.md); [configuration](runtime/configuration.md) | After access is authorized and a supported credential is available, download the exact pinned revision and capture a real inference receipt. |

### User interface and test infrastructure

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ui.settings` | `VALIDATED` | Settings route navigation, four public controls, save/reload/reset, explicit states, responsive layout, and rendered restart readback of the database-backed seed. | [2026-09-28 S50/S52 summary](../QA/validation_campaign/s50-s52-20260928/summary-20260928.md); [restored Settings screenshot](../QA/validation_campaign/s50-s52-20260928/s50-restored-settings.png); [settings E2E](../../app/tests/e2e/test_angular_ui.py); [2026-09-17 E2E note](../QA/e2e-validation-20260917.md); [responsive screenshots](../QA/xreport-settings-narrow-dark.png) | No current issue recorded for the exercised settings surface. | — | 2026-09-28 | E2E + manual rendered browser | [UI experience](ui/experience.md); [UI patterns](ui/components_and_patterns.md) | Revalidate persistence and responsive states after route, contract, or token changes. |
| `ui.inference.catalogue_and_sections` | `VALIDATED` | Rendered inference route, model-card grid, banner behavior, responsive layout, model-declared report sections, model-dependent context controls, profile changes, and desktop catalogue/details geometry. | [S30–S32 campaign](../QA/validation_campaign/s30-s32-20260924/summary.md); [browser observations](../QA/validation_campaign/s30-s32-20260924/browser-observations.json); [catalogue screenshots](../QA/validation_campaign/s30-s32-20260924/inference-catalogue-20260924.jpg); [S33/S52 summary](../QA/validation_campaign/s33-s52-20260924/summary.md); [CheXOne UI contract test](../../app/tests/e2e/test_angular_ui.py) | The catalogue/layout scope is validated. Full generation/review UI evidence currently covers only CXRMate Multi and one public image; other provider output contracts are separate. | — | 2026-09-24 | API E2E + manual rendered review | [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Revalidate provider-declared sections against each relevant model's rendered generation flow. |
| `ui.inference.workflow` | `PARTIAL` | Rendered one-model Generate/poll/cancel/retry/review/provenance/history/edit/copy/export flow plus focused inference-route keyboard, modal, responsive, and reduced-motion checks. | [2026-09-28 S50/S52 summary](../QA/validation_campaign/s50-s52-20260928/summary-20260928.md); [structured receipt](../QA/validation_campaign/s50-s52-20260928/s50-s52-validation-receipt.json); [matrix screenshots](../QA/validation_campaign/s50-s52-20260928/); [S33/S52 summary](../QA/validation_campaign/s33-s52-20260924/summary.md); [browser observations](../QA/validation_campaign/s33-s52-20260924/ui-observations.json); [inference regression tests](../../app/client/src/app/pages/job-cancellation.pages.spec.ts); [modal focus regression test](../../app/client/src/app/components/modal-focus.directive.spec.ts); [production build](../QA/validation_campaign/s33-s52-20260924/client-production-build.log) | One provider and one public image only. The 25-case matrix passed layout, modal, focus, Escape, reduced-motion, screenshot, and browser-error checks; spoken Narrator/Speech Recap output was not observed. The browser download event was not surfaced in this slice, though the expected exported file was previously present and its contents/hash checked. | — | 2026-09-28 | client unit + production build + manual browser E2E | [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Expand coverage to other model output contracts and spoken accessibility observation when the required screen-reader evidence path is available. |
| `ui.reports.history` | `VALIDATED` | Reports navigation, filtered/paginated history cards, persisted detail metadata, section-aware draft editor, explicit empty/loading/error states, edit/reload/delete behavior, and rendered post-restore history readback. | [2026-09-28 S50/S52 summary](../QA/validation_campaign/s50-s52-20260928/summary-20260928.md); [restored Reports screenshot](../QA/validation_campaign/s50-s52-20260928/s50-restored-reports.png); [2026-09-22 reports validation](../QA/validation_campaign/reports-history-validation-20260922.md); [Reports route E2E](../../app/tests/e2e/test_angular_ui.py); [section helper unit tests](../../app/client/src/app/common/report-sections.spec.ts) | The CRUD browser evidence uses deterministic disposable fixtures; clinical review quality and provider-wide generation remain outside this surface. | — | 2026-09-28 | unit + E2E + manual rendered browser | [UI experience](ui/experience.md); [UI patterns](ui/components_and_patterns.md) | Revalidate responsive history behavior after route or editor changes. |
| `ui.startup_gate` | `VALIDATED` | Shell-level startup surface, serialized `/api/health` readiness polling, route suppression before readiness, controlled slow/unavailable/retry recovery, deterministic source-backend schema-drift failure state, ready transition, ordinary post-ready feature error, responsive composition, and current built-bundle rendering. | [2026-09-28 S50/S52 summary](../QA/validation_campaign/s50-s52-20260928/summary-20260928.md); [unavailable screenshot](../QA/validation_campaign/s50-s52-20260928/s50-corrupt-unavailable.png); [startup screenshots](../QA/validation_campaign/s50-s52-20260928/); [2026-09-24 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260924.md); [startup E2E](../../app/tests/e2e/test_angular_ui.py); [frontend unit suite](../../app/client/src/app/services/startup-readiness.service.spec.ts) | Slow/unavailable timing uses a controlled browser clock for the deterministic UI path, while S50 additionally observed the real frontend against a failed source backend. Packaged CPU/CUDA startup and real inference are not covered. | — | 2026-09-28 | unit + E2E + manual rendered browser | [startup](runtime/startup.md); [UI patterns](ui/components_and_patterns.md); [UI experience](ui/experience.md) | Recheck packaged CPU/CUDA startup after the next desktop validation run. |
| `ui.shell.routing_theme_guidance` | `VALIDATED` | Routed shell surfaces, root redirect, direct-load/refresh behavior, exact active navigation for parameterized child routes, Light/Dark/System theme resolution and persistence, versioned guidance dismissal/completion, and manual replay. | [2026-09-22 S10 summary](../QA/validation_campaign/s10/summary-20260922.md); [S10 E2E coverage](../../app/tests/e2e/test_angular_ui.py); [route contract](../../app/client/src/app/app.routes.ts) | The S10 browser gate used disposable report and browser-scoped validation fixtures; complete dataset/training workflows remain outside this status. | — | 2026-09-22 | E2E + manual | [UI experience](ui/experience.md); [UI patterns](ui/components_and_patterns.md) | Revalidate after route, shell, theme, or guidance contract changes. |
| `ui.dataset_and_training_surfaces` | `PARTIAL` | Dataset and Training route rendering; dataset table source-path containment and title; rendered eight-image viewer navigation; processed dataset/checkpoint metadata; S23 dependent-delete conflict and processed-first cleanup; S28 evaluation entry/report review; and the 25-case S52 direct-route responsive/reduced-motion matrix. | [2026-09-28 S23 browser deletion summary](../QA/validation_campaign/s23-delete-ui-20260928/summary-20260928.md); [2026-09-28 S50/S52 summary](../QA/validation_campaign/s50-s52-20260928/summary-20260928.md); [matrix screenshots](../QA/validation_campaign/s50-s52-20260928/); [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md); [S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md); [S27 summary](../QA/validation_campaign/s27/summary-20260923.md); [S28/S34/S53 summary](../QA/validation_campaign/s28-release-20260925/summary.md); [row/viewer captures](../QA/validation_campaign/s23-s26/s23-ui-observations.json); [dataset workflow E2E](../../app/tests/e2e/test_dataset_workflow_ui.py) | The S23 conflict and processed-first order are now recaptured in the browser. Broader dataset lifecycle, representative-data behavior, packaged/native desktop behavior, and spoken screen-reader output remain outside this scope. | — | 2026-09-28 | E2E + manual + live technical workflow | [UI patterns](ui/components_and_patterns.md); [workflows](operations/workflows.md) | Expand only with representative fixtures or a separate packaged/accessibility scope. |
| `ui.validation.report_review` | `VALIDATED` | Successful synthetic dataset-validation and checkpoint-evaluation report completion/review, including persisted timestamps/metrics and reload persistence; missing-data error path remains covered by the existing E2E. | [S27 summary](../QA/validation_campaign/s27/summary-20260923.md); [persisted report receipt](../QA/validation_campaign/s27/persisted-report-response.json); [S28/S34/S53 summary](../QA/validation_campaign/s28-release-20260925/summary.md); [workflow receipts](../QA/validation_campaign/s28-release-20260925/workflow-api-receipts.json); [dataset page regression test](../../app/client/src/app/pages/dataset.page.spec.ts); [missing-dataset E2E](../../app/tests/e2e/test_angular_ui.py) | Eight-row synthetic technical scope only; standalone browser screenshot was not exported. No clinical, representative-data, scale, or report-quality claim. | — | 2026-09-25 | client unit + API + manual rendered review | [UI experience](ui/experience.md); [workflows](operations/workflows.md) | Revalidate after route/report changes; broaden only with an explicit representative-data scope. |
| `test.infrastructure.windows_cache` | `PARTIAL` | Repeatable Windows test startup and disposable-cache routing. | [2026-09-28 S50/S52 summary](../QA/validation_campaign/s50-s52-20260928/summary-20260928.md); [2026-09-24 full-run summary](../QA/validation_campaign/issue-004-windows-cache-20260924/summary.md); [final runner log](../QA/validation_campaign/issue-004-windows-cache-20260924/full-run-after-s15-test-fix.log); [testing rules](coding/testing_and_quality.md) | The focused backend and browser validations used writable disposable basetemp/cache paths because protected historical paths remain ACL-inaccessible. The disposable root was removed and no protected cache path was changed; Git still reports access warnings outside that root. | Ownership/access and safe disposition of historical protected cache paths remain unverified; the configured runner is validated. | 2026-09-28 | focused backend + client unit + browser E2E + manual | [startup](runtime/startup.md); [commands](operations/commands_and_locations.md) | Resolve access/ownership for the historical paths before attempting cleanup; keep using validated isolated runtime caches. |

## Open Issues

Severity is deliberately separate from functional status: a `PARTIAL` component
may have a `LOW` issue, and a `BLOCKED` component may have no software defect.
Only actionable current problems belong here.

| ID | Affected Component | Severity | Concise Description | Current Impact | Reproduction or Evidence | Suspected Cause | Blocker | Remediation Status | Required Revalidation | Related Documentation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ISSUE-001` | `inference.model.cxrmate-ed` | `HIGH` | The exact pinned three-case sensitivity canary completed but produced only two distinct full report texts, so the catalogue remains `degraded`. | CXRMate-ED remains selectable only as an unverified research draft; quality promotion is not justified. | [2026-09-26 S35 receipt](../QA/validation_campaign/s35-s36-20260926/s35-cxrmate-ed-technical-receipt.json): `completed_cases=3`, `reports_all_distinct=false`, `unique_report_count=2`; [case manifest](../QA/validation_campaign/s35-s36-20260926/case-manifest.json) records the exact approved hashes, contexts, and profiles; [manual review](../QA/validation_campaign/s35-s36-20260926/manual-review.json) records the unresolved lateral/normal collapse and differing input provenance. | Not established by the evidence; the differing bytes/dimensions/provenance rule out the previously suspected routing/preprocessing defect. | — | `OPEN` — preserve the degraded warning and do not promote the manifest. | Investigate duplicate outputs across distinct fixtures, then rerun and review all outputs against this exact quality gate. | [local inference models](runtime/local_inference_models.md); [canary command](operations/commands_and_locations.md) |
| `ISSUE-002` | `inference.model.medgemma` | `MEDIUM` | The gated MedGemma entry cannot be installed or generated without authorized provider access and a supported credential. | One of the five catalogue entries is unavailable in the current environment. | [current recheck](../QA/validation_campaign/s33-s52-20260924/gate-recheck.json): `access_policy=gated`, `not_installed`, no `HF_TOKEN` in process/settings; stored credential store not probed; no install/generation attempted. | Not a confirmed software defect. | Authorized gated access and a supported credential. | `BLOCKED` — waiting for authorized access configuration. | Download the exact pinned revision, run real inference, and capture a receipt before changing status. | [gated access](runtime/local_inference_models.md); [configuration](runtime/configuration.md) |
| `ISSUE-006` | `inference.model.cxrmate-multi` | `MEDIUM` | The canonical CXRMate Multi TF resource does not fully match its pinned S31 snapshot manifest: `modelling_multi.py` is 18,493 bytes / SHA-256 `4b06e742bbd3f27b6fd61ef0adfcc693ab7b944fc25b41cc5c6cddc45459f5a9`, expected 17,211 bytes / SHA-256 `528e8d471658802c0ba40fd37a8839cfcd29c342179adeb412f22e04bb7856ee`. | Seven of eight files match. The isolated exact-pinned install and S33 generation passed, but canonical resource integrity and behavior are not established. | [canonical model comparison](../QA/validation_campaign/s33-s52-20260924/canonical-model-recheck.json); [S33/S52 observations](../QA/validation_campaign/s33-s52-20260924/ui-observations.json) | Origin of the differing canonical file is unknown. | Canonical user resource was preserved without modification; use the supported pinned repair workflow before claiming canonical integrity. | `OPEN` — no canonical model file was changed during this validation. | Repair from the exact pinned source, recheck all eight hashes, then repeat canonical load and inference. | [local inference models](runtime/local_inference_models.md) |
| `ISSUE-004` | `test.infrastructure.windows_cache` | `LOW` | Legacy unconfigured pytest runs warned on protected cache paths. The full official `run_tests.bat` now passes through configured runtime caches without pytest cache warnings. | Test execution is validated; Git directory scans still warn when they encounter some pre-existing protected cache paths outside the configured root. | [2026-09-24 runner summary and logs](../QA/validation_campaign/issue-004-windows-cache-20260924/summary.md); [S30–S32 campaign](../QA/validation_campaign/s30-s32-20260924/summary.md); earlier [settings QA](../QA/settings-migration-validation-20260917.md), [E2E QA](../QA/e2e-validation-20260917.md), and [S12 summary](../QA/validation_campaign/s12/summary-20260922.md). | Historical cache directory ACL/ownership, not the official runner configuration. | Filesystem permission/ownership of legacy cache paths. | `OPEN` — official runner is validated; legacy protected paths remain inaccessible and were left unchanged. | Verify ownership/access and safe disposition of the historical paths before cleanup; do not remove cache residue without confirmed ownership. | [testing rules](coding/testing_and_quality.md); [startup](runtime/startup.md) |

## Resolved Issues

| ID | Affected Component | Resolution and Revalidation | Evidence |
| --- | --- | --- | --- |
| `ISSUE-003` | `persistence.checkpoint_registry` | The checkpoint E2E fixture wrote an empty session history, causing complete-artifact validation to warn and making API cleanup ineffective; its fixture now contains one epoch of loss/validation history, and cleanup always reconciles registry and artifact state. Isolated checkpoint listing after the suite was empty with zero matching stale registrations or warnings. | [S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md); [root-cause receipt](../QA/validation_campaign/s23-s26/issue-003.json); [training API tests](../../app/tests/e2e/test_training_api.py) |
| `ISSUE-005` | `workflow.dataset_upload_and_preparation` | Source deletion now checks dependent processing runs within the deletion transaction and returns HTTP 409 with dependent dataset names. The source, records, and training samples remain usable until the processed dependents are deleted; processed-then-source deletion finishes with clean foreign-key and SQLite integrity checks. The current browser flow now shows the same conflict and completes the required processed-first order. | [S23 browser deletion summary](../QA/validation_campaign/s23-delete-ui-20260928/summary-20260928.md); [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md); [deletion regression test](../../app/tests/unit/test_dataset_deletion.py) |

## Validation Debt

Validation debt is not a defect list. It identifies important behavior whose
current confidence is too low or too narrow to support a stronger status.

| Component | Current Confidence | Missing Validation | Priority |
| --- | --- | --- | --- |
| `runtime.desktop.packaged` | Low | CUDA portable/MSI build, NVIDIA execution, provenance, and CPU/CUDA variant separation are now validated for the recorded technical scope. Native WebView navigation, no-GPU fallback, representative performance, and cross-variant benchmark equivalence remain unvalidated. | High |
| `workflow.checkpoint_evaluation` | Medium | Representative-data, scale, separate artifact-file, and packaged-release evaluation evidence beyond the passed eight-row synthetic workflow. | Medium |
| `workflow.inference.generation_pipeline` | Medium | Other-provider browser flows, timeout and persistence-failure behavior, broader reuse conditions, and multi-case quality. | High |
| `inference.model.*` quality | Low | Broader labelled-reference and representative multi-case quality studies remain outstanding for all public models; the current S35 CXRMate-ED gate is failed while S36A CheXOne and S36B CXRMate-2 passed only their conservative technical/output review. | High |
| `ui.inference.workflow` | Medium | Other-provider output contracts, spoken screen-reader observation, browser download-event observation, and provider-wide rendered acceptance remain. | High |
| `launcher.maintenance` | Low | Disposable ClearCache/data/release helper paths, an actual `/T /F` KillProcesses termination of a repo-scoped Windows process tree, and current source FastAPI/Node service-owned listener termination now have evidence. Packaged/Tauri service ownership and canonical release cleanup remain unvalidated. | Medium |

## Resolved / Historical Findings

These entries are retained only to prevent obsolete findings from being
mistaken for active issues. They do not change the current ledger status.

| ID | Component | Previous Finding | Resolution Evidence | Current Residual or Follow-up |
| --- | --- | --- | --- | --- |
| `HIST-001` | `inference.model.chexone` | The 2026-09-19 five-model aggregate rejected CheXOne because it expected incomplete report sections. | [aggregate run](../QA/inference_validation_runs/public-inference-models-20260919T165746Z.json) records the old failure; the [historical 2026-09-20 receipt](../QA/validation_campaign/s35-s36-20260926/historical-chexone-single-case-receipt.json) is retained, and the [2026-09-26 three-case receipt](../QA/validation_campaign/s35-s36-20260926/s36a-chexone-aggregate-technical-receipt.json) passed with non-empty Findings-only output and reload/reuse evidence. | The old aggregate limitation is superseded for the current pinned revision: `validation_status=passed` after the technical and conservative review gates, while broader labelled-reference quality remains unmeasured. |
| `HIST-002` | `runtime.desktop.toolchain` | A settings-migration QA note recorded a pre-existing Tauri capability error during `cargo check`. | The later [2026-09-17 E2E note](../QA/e2e-validation-20260917.md) records `cargo check` passed. | CUDA packaging and native WebView coverage remain validation debt under `runtime.desktop.packaged`. |
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
