# XREPORT Validation Campaign Ledger

Last updated: 2026-09-24

This is the subordinate slice ledger for the long-term XREPORT validation
campaign. [`project_status_ledger.md`](project_status_ledger.md) remains the
canonical aggregate status catalog; this document records slice order,
prerequisites, evidence boundaries, and campaign progress. The campaign is
technical/software validation for research-use workflows. It does not establish
clinical validity, diagnostic performance, or regulatory approval.

## Campaign Rules

- A feature existing, having an automated test, being exercised, and passing
  when observed are four separate facts. Keep them separate in every row.
- Every result is tied to an exact revision, environment, fixture, scenario set,
  and durable evidence path.
- `PASS` is scoped to the stated slice. A technical model receipt is not a
  quality study, and a route-render test is not a complete workflow proof.
- A red hard gate stops downstream claims. Do not spend campaign effort on
  dependent workflows while the current baseline is red.
- Use the sequence **inspect → execute → observe → diagnose → surgically fix →
  retest → record**. Capture the original failure before changing code and run
  the adjacent regression set after a fix.
- Keep disposable databases, model/tool caches, pytest state, and generated
  bundles under `runtimes/cache`. Keep reusable summaries and selected evidence
  under `assets/QA/validation_campaign/` so another checkout can retrieve them.
- Public-model evidence must identify the exact model revision, fixture
  provenance and SHA-256, de-identification statement, requested/actual device,
  output contract, timing, memory, and warning state.

## Gate Model

| Tier | Gate | Slices | Entry/exit rule |
| ---: | --- | --- | --- |
| 0 | Foundational health | `S00 → S01 → S03 → S02` | Current CI baseline, source readiness, cheap deterministic gates, and rendered startup/recovery must be green before feature validation. |
| 1 | State and infrastructure | `S10 → S15` | Shell state, settings, jobs, SQLite/PostgreSQL, and filesystem controls are established. `S10` may follow S02. |
| 2 | Dataset and train/evaluate backbone | `S20 → S28` | A real small dataset and real minimal checkpoint exist before evaluation or custom inference claims. |
| 3 | Inference | `S30 → S35` | Catalogue separation, installation, input validation, browser generation, custom inference, and the degraded-provider canary are explicit. |
| 4 | Heavy providers and desktop | `S36A/S36B → S37` and `S40 → S43` | Heavy/gated providers and each advertised desktop variant have separate evidence; unavailable hardware or access remains explicitly blocked. |
| 5 | Resilience and baseline | `S50 → S53` | Restart/corruption, contention, accessibility/responsiveness, and performance are run only after representative workflows are stable. |

The dependency chain is:

`current CI stability → startup/database → settings/jobs/persistence → dataset → training/checkpoints → validation/custom inference → public inference → desktop packaging → resilience`

## Current Campaign State

- Initial local launch timing and port-preflight evidence came from a Windows `develop` working tree based on `481605b1b87035b8deb03edaefdbfc090f8f1b23`; its uncommitted runtime source was not identified by that base SHA alone. The dated summary retains this scope separately from the current-revision recheck.
- Current application source and validation/test revision: `ad819fdce96bfd237b1eab5586023eb68d932bab` (`develop`). Hosted run `35865126628` passed all configured gates on the prior validation/test revision `2c1825645f7dffbbd669cfa58f7ced5ebc783bdc`, including client unit tests. It resolves the repeated desktop-dialog failures from runs `35862434438` and `35860925179`; the test now stubs the Tauri runtime bridge and exercises the installed Tauri APIs. The Windows focused spec passed (`3 passed`) and full client unit suite passed (`39 passed`). S00 and S03 are `PASS` at that Tier 0 gate scope. Earlier same-code revision `16794b4` also passed all stages on rerun `35859367232`. S20's focused API/service suite passed locally (`12 passed`); see the [S20 summary](../QA/validation_campaign/s20/summary-20260923.md). S21 and S22 now pass on the current revision; see their dated summaries below. S02 remains covered by the [2026-09-23 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260923.md).
- Hosted diagnostic run `35743413087` exposed that the concurrency test depended on an ambient migrated database; its child queried the default resources path before the test initialized the schema, and a clean worker failed with `no such table: application_settings`. The test now creates an isolated SQLite resources directory, initializes its schema before concurrent service creation, and performs ten fresh-process repetitions while retaining the lazy-ML assertions. No production code change was made.
- Windows revalidation on validation/test revision `906b455` passed the complete backend unit suite (`135 passed`), Pyright, and the built-server tests (`9 passed`). Ruff passed with warnings from protected pre-existing cache directories. The initial built-server check observed Windows `ECONNRESET` for the unavailable-backend path; its test now asserts the stable 502 response and accepts `ECONNREFUSED` or `ECONNRESET`.
- A separate normal source-launch and rendered-readiness check was repeated from the committed `3f180d229278aff379358a9f8d3eddf3b07dee5c` checkout. The launcher passed port, dependency, and build checks; it started the backend and frontend and reported the UI URL, then its automatic Windows browser-open step returned `Access denied`. Opening the URL in the Codex in-app browser showed the ready inference workspace and model catalogue; both health URLs returned HTTP 200, and the exact server processes were stopped afterward. Hosted CI did not execute this Windows launcher or rendered-browser flow. The original launch/port-preflight scenarios and remaining edge cases are distinguished in [the 2026-09-22 execution summary](../QA/validation_campaign/tier-0/summary-20260922.md).
- S10 passed on 2026-09-22 from `develop` HEAD `23f56f489260367db83e16e5925e1a562d175779` plus the reviewed working-tree test/launcher fixes. The browser matrix covered the redirect, every routed surface and refresh, parameterized report and dataset-validation fixtures, active navigation, Light/Dark/System persistence and media changes, guidance dismiss/skip/complete/replay, screenshots, and console/request cleanliness. The durable [S10 summary](../QA/validation_campaign/s10/summary-20260922.md) records the exact environment and the adjacent Angular build/toolchain limitation.
- S02 passed on 2026-09-23 against current application revision `fdb4a8b`: the browser observed two unhealthy health responses followed by ready state, inspected the current startup artwork, and captured both loading and ready screenshots. The new evidence is in the [2026-09-23 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260923.md); explicit slow/unavailable/retry and post-ready feature-error scenarios remain open.
- S14 passed on 2026-09-23 against revision `f6ee0a9` in the hosted PostgreSQL 16 service. The application initializer applied Alembic migrations to head, persisted settings across reinitialization, completed concurrent first-time initialization under the PostgreSQL advisory locks, and kept credentials out of a real refused-connection error and its logs. The first run exposed false schema-drift reports for PostgreSQL-rendered constraints and unique-constraint indexes; the comparison was corrected and covered by unit regressions. See the [S14 summary](../QA/validation_campaign/s14/summary-20260923.md).
- S15 passed on 2026-09-23 against application source revision `0ace867ea8087a112305bb9acce1770f259e2f2c`. Focused API and Angular coverage passed; the official Windows launcher and rendered Dataset workflow covered filesystem access disabled/enabled, invalid path feedback and recovery, empty-folder feedback, and successful one-image selection. The launcher used an isolated resources directory and synthetic fixtures. No application source change was needed. The subsequent S21/S22 evidence completes the broader dataset preparation workflow. See the [S15 summary](../QA/validation_campaign/s15/summary-20260923.md).
- S20 passed on 2026-09-23 against committed source/test revision `37a4b78e6ca939f8a2b6cb9e29c8fa7d46a3d41e`. The focused API/service suite covered comma and semicolon CSV, generated XLSX, invalid/empty input, unsupported extensions, the 16 MiB limit, independent upload contents, and unknown upload IDs. This remains parsing and upload-identity evidence only; image matching, processing, and persistence are covered by S21/S22 below. See the [S20 summary](../QA/validation_campaign/s20/summary-20260923.md).
- S21 and S22 passed on application and validation/test revision `ad819fdce96bfd237b1eab5586023eb68d932bab` (`develop`). S21 covered eight-row full match, case-insensitive stems, partial preview with no pre-confirmation writes, rendered confirmation/counts, no-match rejection, required-column errors, and persisted source metadata. S22 covered the minimum-one sampling regression, an eight-row DistilBERT processing run at 50% and a same-name repeat at 100%, 3/1 and 6/2 train/validation counts, missing-image failure, two processing-run history records, and source/processed metadata plus SQLite integrity after backend restart. The local suite passed (10 focused service tests, 1 sampling regression, 3 API/UI E2E tests, Ruff, and `git diff --check`). Hosted CI run [35880057345](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35880057345) passed every configured gate on ledger revision `a4e83cd619f9514cdf1c384a8ce47cacd16eae31`, including the source/test revision. The UI captures, fixture hashes, tokenizer revision, and remaining scope limits are recorded in the [S21 summary](../QA/validation_campaign/s21/summary-20260923.md) and [S22 summary](../QA/validation_campaign/s22/summary-20260923.md).
- S23–S26 were exercised on 2026-09-23 from a clean `develop` baseline at `3846609cc68646b501cd480c2a11938fa2278f8b`, using an isolated SQLite resource root, eight synthetic rows, the pinned DistilBERT tokenizer snapshot, the pinned BEiT encoder, and an RTX 3060. The exact validated changes were committed as `d665ee078ef5a878113f3db03c518834b5afdd68`; hosted CI [35904882475](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35904882475) passed every configured backend and client check. S23's 2026-09-23 deletion check was `PARTIAL`: source deletion cascaded source records and emptied the surviving processed dataset. The 2026-09-24 follow-up now passes the safe-delete contract: the API returns 409 with dependent processed dataset names, preserves its rows and training samples, and permits deletion in processed-then-source order with SQLite integrity clean. See the [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md) and the [original S23–S26 summary](../QA/validation_campaign/s23-s26/summary-20260923.md). S24 passed a real one-epoch CUDA training job; S25 passed cancellation after progress, worker exit, and resume to two persisted epochs; S26 passed ready listing/metadata, referenced/unsafe deletion guards, and safe artifact plus registry removal. The campaign exposed and fixed process-level `XREPORT_RESOURCES_DIR` loss during dotenv loading, a fixture-history root cause for `ISSUE-003`, and the source-path table overflow reported by the user. The focused dataset/training E2E suite passed (`8 passed`), the environment regression passed (`3 passed`), Ruff and frontend build passed. Synthetic training is technical-path evidence only, not a clinical or performance claim.
- S27 passed on 2026-09-23 from source baseline `40ac5c84b953861ce3f1c2cf7657c27032bf298f` using a disposable SQLite root and eight generated synthetic rows/images. Full-dataset validation completed twice with all three metric families; both jobs succeeded, the report persisted and returned over HTTP 200, and the rebuilt route displayed the saved timestamp. Backend focused checks passed (`22`), the full client unit suite passed (`40`), and the production frontend rebuild plus browser review passed. The completion view initially showed `Generated: N/A`; its metadata refresh was fixed and covered by a regression test. See the [S27 summary](../QA/validation_campaign/s27/summary-20260923.md), [fixture manifest](../QA/validation_campaign/s27/fixture-manifest.json), and [API receipts](../QA/validation_campaign/s27/).

## Tier 0 Slice Ledger — Current Execution

| Slice | Capability | Exists | Exercised | Status | Scenarios and regressions | Evidence | Remaining gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `S00` | Restore current CI baseline | yes | yes | `PASS` | Hosted run `35865126628` passed every configured gate on validation/test revision `2c18256`, including the repaired desktop-dialog client unit tests. The Windows focused spec passed (`3 passed`) and complete client suite passed (`39 passed`). Previous failures on revisions `caa4a41` and `5cbe2a7` remain historical; the earlier same-code revision `16794b4` also passed all stages on run `35859367232` after rerun. | [2026-09-23 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260923.md); [green hosted CI run 35865126628](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35865126628); [previous failed run 35862434438](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35862434438) | S21–S27 are recorded in their own workflow rows; run the hosted baseline on the intended release revision before release. |
| `S01` | Source backend startup and SQLite readiness | yes | yes | `PASS` | On committed code/test SHA `3f180d2`, the Windows launcher passed port/dependency/build checks, started FastAPI and the built UI, and reported the UI URL; the browser auto-open returned `Access denied`, after which the local URL was opened manually and the ready workspace/catalogue loaded. Both health URLs returned HTTP 200. | [2026-09-22 execution summary](../QA/validation_campaign/tier-0/summary-20260922.md) | Complete the still-open launcher port-race/failure and build dependency-invalidation scenarios before broadening the source-startup claim. |
| `S02` | Built frontend startup gate and backend recovery | yes | yes | `PASS` | On current application revision `fdb4a8b`, the focused gate test observed two `503` health responses then `200`, verified startup image decoding and route suppression, and captured the current loading screen plus the ready workspace in the in-app browser. Hosted CI on validation revision `906b455` also passed built-server checks. | [2026-09-23 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260923.md); [CI run 35834323975](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35834323975) | Explicit slow/unavailable/retry timing and post-ready ordinary feature-error evidence remain open. |
| `S03` | Static quality, API contract, and client build gates | yes | yes | `PASS` | Hosted run `35865126628` on validation/test revision `2c18256` passed Ruff, Pyright, backend unit tests, PostgreSQL contract, API E2E, client build/server, lint, and client unit tests. The focused Windows client spec passed (`3 passed`) and complete suite passed (`39 passed`). Current-scope local production build and rendered checks remain in the dated execution summary. | [2026-09-23 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260923.md); [green hosted CI run 35865126628](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35865126628); [2026-09-22 execution summary](../QA/validation_campaign/tier-0/summary-20260922.md) | Dataset and inference workflows are tracked in separate slices; this row covers only foundational static and client gates. |

### Tier 0 Evidence Notes

The 2026-09-21 launcher attempt to open the URL through Windows `Start-Process`
returned access denied in the managed environment. An earlier 2026-09-22 run
completed its browser-open step on the then-current working tree. A follow-up
from committed code/test SHA `3f180d2` again reached backend and frontend
readiness, but its automatic browser-open step returned access denied; the URL
was opened manually in the Codex in-app browser and rendered the ready
workspace. The follow-up processes were stopped by verified paths and ports
`5003` and `8003` were confirmed clear. Port-guard cases not exercised are
enumerated in the dated summary.

On 2026-09-21, an `npm ci` attempt encountered `EPERM` unlinking the launch-held
`esbuild.exe`; after the verified XREPORT frontend tree was stopped, `npm ci`
completed successfully. This is historical cleanup evidence related to the
Windows cache/process issue, not a client build defect. No `npm ci` was needed
or run for the 2026-09-22 change set because dependency manifests were unchanged.

## Remaining Slice Ledger

These rows are the campaign backlog. They preserve the stable slice IDs and
the expected next boundary without claiming that implementation or focused
tests constitute workflow validation.

### Tier 1 — State and Infrastructure

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S10` | Routing, navigation, theme, and guidance persistence | `S02`, `S03` | `PASS` | Direct-load/refresh for every route, active navigation including parameterized child routes, all theme modes, persisted theme/guidance state, manual guidance replay, and console/request cleanliness passed in the Windows in-app browser run. [2026-09-22 S10 summary](../QA/validation_campaign/s10/summary-20260922.md); [S10 E2E coverage](../../app/tests/e2e/test_angular_ui.py) |
| `S11` | Runtime settings persistence, limits, reset, and hidden-field protection | `S01` | `VALIDATED` for current exercised surface | GET/PATCH/reset payloads, database row before/after, restart persistence, bounds, and captured-value behavior for active jobs. |
| `S12` | Generic job lifecycle | `S01` | `PASS` | Deterministic API coverage passed for start/list/type-and-status filtering/poll/complete, unknown ID, active cancellation through terminal state, and typed recoverable failure. Existing client polling tests passed for transient/repeated transport failures and missing-job termination. [2026-09-22 S12 summary](../QA/validation_campaign/s12/summary-20260922.md); [API lifecycle tests](../../app/tests/integration/test_job_lifecycle_api.py); [polling tests](../../app/client/src/app/services/job-polling.service.spec.ts). |
| `S13` | SQLite migration and restart persistence | `S01` | `PASS` | Fresh isolated SQLite reached Alembic head `e91a4f6c2d73`; a non-default setting and synthetic inference history with linked report survived backend stop/relaunch and were read back through the API. Existing tests passed for unversioned/unknown schema rejection, migration rollback, and schema drift; the new focused test proves FK enforcement, orphan rejection, and child cascade. | [2026-09-23 S13 summary](../QA/validation_campaign/s13/summary-20260923.md); [database initialization tests](../../app/tests/unit/test_database_initialization.py); [settings tests](../../app/tests/unit/test_application_settings.py); [repository persistence tests](../../app/tests/unit/test_repository_persistence.py); [hosted CI run 35834323975](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35834323975) |
| `S14` | PostgreSQL persistence contract | `S00` | `PASS` | Hosted PostgreSQL 16 validation applied application migrations to Alembic head `e91a4f6c2d73`, verified the schema contract and settings persistence across reinitialization, passed concurrent first-time initialization under the creation and migration advisory locks, and confirmed a refused connection did not expose credentials. The full hosted CI run also passed. [2026-09-23 S14 summary](../QA/validation_campaign/s14/summary-20260923.md); [integration tests](../../app/tests/integration/test_persistence_contract.py); [CI run 35841152401](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35841152401) |
| `S15` | Local filesystem access and path controls | `S11` | `PASS` | API and rendered browser scenarios passed for enabled/disabled access, accessible image folder, invalid/non-folder paths, empty folder, valid image selection with count, and recovery after an invalid path. The [S15 summary](../QA/validation_campaign/s15/summary-20260923.md) records the exact revision and evidence. The overall dataset workflow is still open. |

### Tier 2 — Dataset and Train/Evaluate Backbone

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S20` | CSV/XLSX upload parsing and explicit upload identity | `S10` | `PASS` | API and service checks passed for comma/semicolon CSV, generated XLSX, empty/corrupt and unsupported files, exact and over-limit payload sizes, independent upload IDs/content, and unknown upload ID. [2026-09-23 S20 summary](../QA/validation_campaign/s20/summary-20260923.md) |
| `S21` | Image matching and partial-import confirmation | `S20`, `S15` | `PASS` | Full/partial/no match, confirmation before import, rendered counts, required-column errors, case-insensitive stems, and persisted source rows passed in service, API, and rendered browser checks. [2026-09-23 S21 summary](../QA/validation_campaign/s21/summary-20260923.md) |
| `S22` | Dataset processing and integrity | `S21` | `PASS` | Minimum-one sampling, real tokenizer processing, 50%/100% sample counts, 3/1 and 6/2 train/validation splits, same-name identity with two run-history records, missing-image failure, and restart metadata passed on the isolated eight-row fixture. Functional checks only; no scale/performance claim. [2026-09-23 S22 summary](../QA/validation_campaign/s22/summary-20260923.md) |
| `S23` | Dataset viewer and deletion | `S22` | `PASS` | Rendered long-path row containment and eight-image viewer navigation passed in the original S23 browser run. The 2026-09-24 API regression now returns 409 and names dependent processed datasets while preserving source and training rows; the sample remains loadable, and processed-then-source deletion completes with no foreign-key violations and SQLite integrity `ok`. See [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md), [original UI observations](../QA/validation_campaign/s23-s26/s23-ui-observations.json), and [original deletion receipt](../QA/validation_campaign/s23-s26/s23-deletion.json). Deleting a source now requires deleting processed dependents first; the rejected-delete message was not recaptured in a browser screenshot. |
| `S24` | Minimal real training | `S22` | `PASS` | One real synthetic-dataset epoch completed on CUDA with a ready checkpoint, persisted configuration, provenance, and epoch history. The exact pinned BEiT revision and weight hash are recorded. This is software-path evidence only; no quality or performance claim. [Training receipt](../QA/validation_campaign/s23-s26/s24-s25-training.json); [checkpoint API evidence](../QA/validation_campaign/s23-s26/s26-checkpoint-api.json). |
| `S25` | Training stop and resume | `S24` | `PASS` | A separate 100-epoch CUDA job cancelled at 5%/epoch 5 and reached terminal cancellation with worker exit. The S24 checkpoint then resumed for one epoch; persisted history advanced from one to two epochs. [Training receipt](../QA/validation_campaign/s23-s26/s24-s25-training.json); [checkpoint API evidence](../QA/validation_campaign/s23-s26/s26-checkpoint-api.json). |
| `S26` | Checkpoint registry, metadata, and deletion | `S24` | `PASS` | Ready listing and metadata contained dataset/configuration/history; deletion of a referenced checkpoint returned 409 and retained artifact/registry; unsafe path returned 400; unreferenced deletion removed both artifact and registry. `ISSUE-003` was traced to incomplete test fixture history and corrected; isolated list and database had no stale `e2e_delete_*` registrations or matching warnings. [Checkpoint API evidence](../QA/validation_campaign/s23-s26/s26-checkpoint-api.json); [ISSUE-003 trace](../QA/validation_campaign/s23-s26/issue-003.json); [summary](../QA/validation_campaign/s23-s26/summary-20260923.md). |
| `S27` | Successful dataset validation | `S22` | `PASS` | Full-dataset validation on an eight-row synthetic fixture completed twice with text, image, and pixel-distribution metrics. Both jobs succeeded; the saved report returned HTTP 200 with matching dataset, sample size, metrics, and persisted timestamp; the rebuilt route rendered the report successfully. [S27 summary](../QA/validation_campaign/s27/summary-20260923.md); [fixture manifest](../QA/validation_campaign/s27/fixture-manifest.json); [API receipts and backend log](../QA/validation_campaign/s27/). Synthetic technical workflow only; no clinical, representative-data, or scale claim. The backend returned an empty `artifacts` map, so separate artifact-file generation was not validated. The browser capture was reviewed inline but not exported as a standalone PNG. |
| `S28` | Successful checkpoint evaluation | `S24`, `S27` | `UNTESTED` | Rechecked 2026-09-24: neither campaign checkpoint path contains an artifact, the canonical checkpoint directory contains only `.gitkeep`, and the exact pinned BEiT snapshot is absent from its expected path and Hugging Face cache. The evaluation was not run. Restore the encoder, recreate S24, then evaluate its compatible checkpoint and retrieve the saved report. [2026-09-24 prerequisite probe](../QA/validation_campaign/s28/prerequisite-check-20260924.json); [2026-09-23 probe](../QA/validation_campaign/s27/s28-prerequisite-check.json). |

### Tier 3 — Inference

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S30` | Inference catalogue and public/custom separation | `S26` | `PARTIAL` | Five pinned public entries, exact revisions, readiness state, declared sections, and separate custom checkpoint identity. |
| `S31` | Lightweight public-model installation lifecycle | `S30` | `PARTIAL` | First install, verification, reuse without redownload, repair/delete, and readiness/provenance transitions. |
| `S32` | Inference input and browser-state validation | `S30` | `PARTIAL` | Image limits, supported inputs, clinical-context contract, profile changes, invalid state, and safe typed errors. |
| `S33` | Complete browser inference workflow | `S31`, `S32` | `UNTESTED` | Rendered Generate, poll, cancel, retry, edit, copy, export, provenance, and persistence flow. |
| `S34` | Custom XREPORT checkpoint inference | `S24`, `S26` | `UNTESTED` | Real generated checkpoint selected as custom, generated report, provenance, and report persistence. |
| `S35` | CXRMate-ED sensitivity canary | `S31` | `FAIL` / degraded | Three-case quality canary must produce distinct, reviewed outputs before promotion; keep `degraded` until then. |

### Tier 4 — Heavy Providers and Desktop

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S36A` | CheXOne technical and broader quality gate | `S33` | `PARTIAL` | Exact pinned revision, Findings-only contract, multi-case quality evidence, fixture provenance, and resource measurements. |
| `S36B` | CXRMate-2 technical and broader quality gate | `S33` | `PARTIAL` | Exact pinned revision, multi-case quality evidence, resource budget, and reuse/repair behavior. |
| `S37` | MedGemma gated-provider behavior | `S30` | `BLOCKED` | Authorized terms/credential check, exact pinned install, real generation receipt, or explicit continued block. |
| `S40` | Tauri development shell | `S02`, `S03` | `PARTIAL` | Shell startup, backend readiness, navigation, shutdown, and data-root separation. |
| `S41` | CPU packaged desktop | `S40` | `UNTESTED` | CPU portable and MSI build, checksum/manifest, launch, readiness, shutdown, and user-data isolation. |
| `S42` | CUDA packaged desktop | `S41` or compatible hardware | `UNTESTED` | CUDA portable/MSI build, NVIDIA execution, actual-device provenance, fallback behavior, and variant separation. |
| `S43` | Windows launcher maintenance actions | `S13` | `PARTIAL` | No-op and disposable-fixture checks for ClearCache, RemoveCheckpoints, RemoveAllData, RemoveDesktopRelease, and KillProcesses. |

### Tier 5 — Resilience and Baseline

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S50` | Restart and corruption recovery | Tiers 1–3 | `UNTESTED` | Durable-state manifest, missing-image behavior, partial model snapshot, corrupt metadata/database, and fail-closed recovery. |
| `S51` | Concurrent operations and resource contention | Tiers 2–3 | `UNTESTED` | Accepted/rejected overlaps, rapid cancel/restart, UI close/reopen, process list, memory, and no stuck jobs. |
| `S52` | Accessibility and responsive workflow audit | Stable Tiers 2–3 UI | `PARTIAL` | Keyboard/focus/modal behavior, accessible names/status, reduced motion, no blocking overflow at 320px/tablet/1024x720/desktop. |
| `S53` | Performance and long-operation baseline | Stable representative workflows | `UNKNOWN` | Hardware, fixture, revision, startup/processing/training/load/generation timings, RAM/CUDA memory, and packaged startup measurements. |

## Evidence Bundle Contract

For every slice, create a small summary at:

```text
assets/QA/validation_campaign/<tier-or-slice>/summary-YYYYMMDD.md
```

The summary records:

- exact Git revision and dirty-tree state;
- operating system, hardware, runtime variant, and database backend;
- exact commands and scenario IDs executed;
- `PASS`, `PARTIAL`, `FAIL`, `BLOCKED`, `UNTESTED`, or `UNKNOWN` result;
- issue references, fixes, and adjacent regression results;
- durable paths or hashes for screenshots, logs, responses, receipts, and
  measurements;
- explicit omissions and the next action.

Raw transient output may remain in `runtimes/cache`, but it must not be the
only proof for a durable `PASS`. Screenshots are required for visible browser or
desktop acceptance; backend slices require request/response and log evidence.

## Regression and Stopping Rules

Run the smallest adjacent regression set after a change: database changes use
S01/settings/list endpoints; settings changes use S11/startup/one new job; job
changes use S12 plus one consumer; upload changes use S20/S21; dataset changes
use S21/S22/S27; training changes use S24/S25/checkpoint listing; catalogue or
provider changes use S30/request validation plus one unaffected lightweight
provider; Angular shell changes use S02 and route smoke; packaging changes use
the affected CPU/CUDA artifact checks.

XREPORT may be described as comprehensively validated only for a defined release
scope when the release SHA has green CI, Tier 0 has no unresolved failure, the
in-scope dataset and training backbone are real and complete, browser inference
is exercised end to end, every in-scope public model has pinned technical
evidence, gated/degraded providers are explicit, advertised desktop variants
have evidence, restart/recovery is covered, no open HIGH issue remains, and
every PASS is tied to durable evidence. The final wording must retain the
research-use limitation.
