# XREPORT Validation Campaign Ledger

Last updated: 2026-10-01

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
- Tier 5 scope is source-runtime technical validation. `S51` covers concurrent
  application workloads, atomic admission, cancellation/failure isolation,
  worker cleanup, job attribution, and resource contention; packaged desktop
  and native WebView behavior remain Tier 4 concerns. `S52` covers inspectable
  browser semantics, accessible names and labels, keyboard/focus behavior,
  modal containment/restoration, responsive reflow, scrolling, reduced motion,
  visible errors, and browser collectors; real Narrator, Speech Recap, or other
  assistive-technology speech output is outside the gate. `S53` covers
  reproducible source-runtime technical performance baselines, including
  scaled workloads, repeated measurements, CPU/CUDA comparisons where
  available, and bounded long operations; packaged timing, native WebView
  performance, no-GPU fallback, and clinical-quality benchmarking are outside
  the gate.

## Gate Model

| Tier | Gate | Slices | Entry/exit rule |
| ---: | --- | --- | --- |
| 0 | Foundational health | `S00 → S01 → S03 → S02` | Current CI baseline, source readiness, cheap deterministic gates, and rendered startup/recovery must be green before feature validation. |
| 1 | State and infrastructure | `S10 → S15` | Shell state, settings, jobs, SQLite/PostgreSQL, and filesystem controls are established. `S10` may follow S02. |
| 2 | Dataset and train/evaluate backbone | `S20 → S28` | A real small dataset and real minimal checkpoint exist before evaluation or custom inference claims. |
| 3 | Inference | `S30 → S35` | Catalogue separation, installation, input validation, browser generation, custom inference, and the public-model qualification/retirement gate are explicit. |
| 4 | Heavy providers and desktop | `S36A/S36B → S37` and `S40 → S43` | Heavy/gated providers and each advertised desktop variant have separate evidence; unavailable hardware or access remains explicitly blocked. |
| 5 | Resilience and baseline | `S50 → S53` | Restart/corruption, contention, accessibility/responsiveness, and performance are run only after representative workflows are stable. |

The dependency chain is:

`current CI stability → startup/database → settings/jobs/persistence → dataset → training/checkpoints → validation/custom inference → public inference → desktop packaging → resilience`

## Current Campaign State

- Tier 0 is green on the current validation revision
  `7bf8402f90bdbf2181f13190782b67ba51d445d9` (`develop`, clean and aligned with
  `origin/develop`): hosted CI run
  [36422997859](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/36422997859)
  passed every configured gate, and the later hosted run `36586991512` for
  commit `064278f8` also passed. `S00`, `S01`, `S02`, and `S03` are `PASS` at
  their recorded gate scope; see the Tier 0 ledger below.
- S11 and S37 closed `PASS` on 2026-10-01 (runtime-settings contract and
  gated-provider behavior); `ISSUE-002` remains open for real gated access.
  S40 (native Tauri shell), S41 (CPU packaged), S43 (launcher maintenance),
  S50, S52, and S53 are `PASS` at their defined scopes. S51 and S42 remain
  `PARTIAL`: S51 still needs genuine GPU-less hardware, hosted-CI,
  representative-data, and slow-initialization attribution; S42 still needs a
  genuine no-GPU package lane.
- The v3.1.0 CPU/CUDA portable/MSI pairs are bound to source `301a5a51` with
  `dirty_tree=false`; both full MSI install → launch → graceful-close →
  uninstall lifecycles passed on 2026-10-01. The staged
  `approved-release-manifest.json` still awaits human approval.
- The full live-service Python suite passed `235` with `3` documented PostgreSQL
  skips. The Angular production builder still crashes with Windows
  `0xC0000005`, so no new full S03 production-build PASS is claimed beyond the
  recorded hosted-CI runs.
- The complete official Windows `app/tests/run_tests.bat` runner passed
  (Python `191`, frontend unit `40`, Angular E2E `14`) through the configured
  runtime caches; only protected legacy cache paths still produce Git access
  warnings (`ISSUE-004`).

## Decisions and Supersessions

- **Scope boundaries.** Tier 5 covers source-runtime technical validation;
  packaged desktop and native WebView behavior remain Tier 4 concerns. Spoken
  Narrator/Speech Recap output is a separate non-gating accessibility follow-up
  and does not lower S52 or any other gate status. `S02` timing is
  controlled-clock E2E evidence, not a wall-clock outage; packaged CPU/CUDA
  startup stays outside it.
- **CXRMate-ED retired (2026-09-29).** The bounded three-case rescue produced
  one identical full report, so `ISSUE-001` was resolved by retirement rather
  than qualification; the earlier `FAIL / degraded` disposition is superseded.
  Historical reports, provenance, and the downloaded snapshot remain preserved.
- **S51 race fix (2026-09-25).** A deterministic harness reproduced a
  check-then-start double admission across training/resume, processing,
  validation, and evaluation. Admission is now atomic, with a regression gate
  and later clean-restart matrices passing.
- **S43 ClearCache fix (2026-09-25).** The launcher passed a single-character
  prefix instead of the full cache path to `Remove-LauncherPath`; the target
  pipeline is now wrapped before indexing in `start_on_windows.ps1`.
- **Resolved issues.** `ISSUE-003` (incomplete checkpoint fixture history),
  `ISSUE-005` (dependent dataset-delete protection via HTTP 409), and
  `ISSUE-006` (canonical CXRMate Multi manifest restored to the S31 snapshot on
  2026-10-01) are resolved.

## Tier 0 Slice Ledger — Current Execution

| Slice | Capability | Exists | Exercised | Status | Scenarios and regressions | Evidence | Remaining gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `S00` | Restore current CI baseline | yes | yes | `PASS` | Hosted run `36422997859` passed every configured gate on exact validation revision `7bf8402f90bdbf2181f13190782b67ba51d445d9`, including backend and client validation plus PostgreSQL persistence. Earlier successful runs `36014118210`, `35865126628`, and `35859367232` and historical failures remain recorded below. | [2026-09-28 hosted CI run 36422997859](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/36422997859); [2026-09-24 hosted CI run 36014118210](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/36014118210); [2026-09-23 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260923.md); [previous failed run 35862434438](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35862434438) | Re-run on the intended release revision if source or test behavior changes before release. |
| `S01` | Source backend startup and SQLite readiness | yes | yes | `PASS` | On current revision `7bf8402`, the official source launcher initialized a fresh isolated process-override root, reached backend and frontend-proxy health `200`, rendered readiness, and initialized SQLite at Alembic head `e91a4f6c2d73`. A saved seed survived ordinary stop/relaunch and rendered in Settings. The fixture harness confirmed default `RepoRoot\data`, repo-relative dotenv resolution, and process-override precedence. | [current data-root summary](../QA/validation_campaign/s01-s43-s50-data-root-20260928/summary-20260928.md); [validation receipt](../QA/validation_campaign/s01-s43-s50-data-root-20260928/validation-receipt.json); [root/maintenance harness receipt](../QA/validation_campaign/s01-s43-s50-data-root-20260928/launcher-data-root-harness-receipt.json); [clean restart receipt](../QA/validation_campaign/s01-s43-s50-data-root-20260928/clean-restart-receipt.json); [current-head S51/S53 summary](../QA/validation_campaign/s51-s53-20260928/summary-20260928.md); [2026-09-24 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260924.md) | Scoped startup and noninteractive occupied-port refusal passed. Overall source startup remains `PARTIAL`: interactive termination/ownership races, output deletion, source-change rebuild, package-lock/npm-ci recovery, and edit-during-build remain untested. |
| `S02` | Built frontend startup gate and backend recovery | yes | yes | `PASS` | The 2026-09-24 focused gate suite passed (`2 passed`): controlled-clock health responses rendered slow and unavailable states, exposed retry, recovered to ready, suppressed routes before health success, and kept a synthetic post-ready catalogue 503 inline without returning to the startup screen. The existing loading/ready behavior was rendered in the in-app browser. | [2026-09-24 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260924.md); [screenshots](../QA/validation_campaign/tier-0/s02-20260924/); [2026-09-23 summary](../QA/validation_campaign/tier-0/summary-20260923.md) | Timing is controlled-clock E2E evidence with intercepted health/catalogue responses, not a wall-clock outage. Packaged CPU/CUDA startup and real inference remain outside this slice. |
| `S03` | Static quality, API contract, and client build gates | yes | yes | `PASS` | Current revision `7bf8402` passed hosted run `36422997859`, including the configured backend, client, and PostgreSQL gates. The broader current-head local recheck on the adjacent campaign revision passed `166` backend unit tests, Ruff, Pyright (`0` errors/warnings/informations), Angular production build, client lint, and `43` client unit tests. | [2026-09-28 hosted CI run 36422997859](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/36422997859); [current-head S51/S53 summary](../QA/validation_campaign/s51-s53-20260928/summary-20260928.md); [2026-09-24 hosted CI run 36014118210](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/36014118210); [2026-09-23 Tier 0 summary](../QA/validation_campaign/tier-0/summary-20260923.md) | Dataset and inference workflows are tracked in separate slices; this row covers only foundational static and client gates. |

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

On 2026-09-24, `Get-NetTCPConnection` returned access denied in the managed
Windows environment. The launcher now falls back to `netstat.exe` to identify
configured listening ports; when process metadata is also unavailable, it
reports PIDs and fails closed without terminating processes in noninteractive
mode. The official launcher also treats a denied automatic browser open as a
warning after both services are ready. Interactive process-ownership races and
dependency-invalidation cases remain open as listed in S01.

## Remaining Slice Ledger

These rows are the campaign backlog. They preserve the stable slice IDs and
the expected next boundary without claiming that implementation or focused
tests constitute workflow validation.

### Tier 1 — State and Infrastructure

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S10` | Routing, navigation, theme, and guidance persistence | `S02`, `S03` | `PASS` | Direct-load/refresh for every route, active navigation including parameterized child routes, all theme modes, persisted theme/guidance state, manual guidance replay, and console/request cleanliness passed in the Windows in-app browser run. [2026-09-22 S10 summary](../QA/validation_campaign/s10/summary-20260922.md); [S10 E2E coverage](../../app/tests/e2e/test_angular_ui.py) |
| `S11` | Runtime settings persistence, limits, reset, and hidden-field protection | `S01` | `PASS` | Live GET/PATCH/reset on a disposable-root SQLite backend: public projection, persisted seed readback, bounds (`polling_interval` `0.25`–`60.0`), hidden-field protection (`inference.device` 422), null rejection, and reset-to-defaults; the backend unit and settings page client specs also passed. [2026-10-01 S11 summary](../QA/validation_campaign/s11/summary-20261001.md); [live receipt](../QA/validation_campaign/s11-s37-20261001/s37-s11-live-receipt.json) |
| `S12` | Generic job lifecycle | `S01` | `PASS` | Deterministic API coverage passed for start/list/type-and-status filtering/poll/complete, unknown ID, active cancellation through terminal state, and typed recoverable failure. Existing client polling tests passed for transient/repeated transport failures and missing-job termination. [2026-09-22 S12 summary](../QA/validation_campaign/s12/summary-20260922.md); [API lifecycle tests](../../app/tests/integration/test_job_lifecycle_api.py); [polling tests](../../app/client/src/app/services/job-polling.service.spec.ts). |
| `S13` | SQLite migration and restart persistence | `S01` | `PASS` | Fresh isolated SQLite reached Alembic head `e91a4f6c2d73`; a non-default setting and synthetic inference history with linked report survived backend stop/relaunch and were read back through the API. Existing tests passed for unversioned/unknown schema rejection, migration rollback, and schema drift; the new focused test proves FK enforcement, orphan rejection, and child cascade. | [2026-09-23 S13 summary](../QA/validation_campaign/s13/summary-20260923.md); [database initialization tests](../../app/tests/unit/test_database_initialization.py); [settings tests](../../app/tests/unit/test_application_settings.py); [repository persistence tests](../../app/tests/unit/test_repository_persistence.py); [hosted CI run 35834323975](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35834323975) |
| `S14` | PostgreSQL persistence contract | `S00` | `PASS` | Hosted PostgreSQL 16 validation applied application migrations to Alembic head `e91a4f6c2d73`, verified the schema contract and settings persistence across reinitialization, passed concurrent first-time initialization under the creation and migration advisory locks, and confirmed a refused connection did not expose credentials. The full hosted CI run also passed. [2026-09-23 S14 summary](../QA/validation_campaign/s14/summary-20260923.md); [integration tests](../../app/tests/integration/test_persistence_contract.py); [CI run 35841152401](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35841152401) |
| `S15` | Local filesystem access and path controls | `S11` | `PASS` | API and rendered browser scenarios passed for enabled/disabled access, accessible image folder, invalid/non-folder paths, empty folder, valid image selection with count, and recovery after an invalid path. The full Windows runner exposed a test-only timing race in the immediate enabled/disabled assertions; retrying Playwright expectations now await the asynchronous page status, and the complete rerun passed. See the [original S15 summary](../QA/validation_campaign/s15/summary-20260923.md) and [ISSUE-004 runner evidence](../QA/validation_campaign/issue-004-windows-cache-20260924/summary.md). The overall dataset workflow is still open. |

### Tier 2 — Dataset and Train/Evaluate Backbone

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S20` | CSV/XLSX upload parsing and explicit upload identity | `S10` | `PASS` | API and service checks passed for comma/semicolon CSV, generated XLSX, empty/corrupt and unsupported files, exact and over-limit payload sizes, independent upload IDs/content, and unknown upload ID. [2026-09-23 S20 summary](../QA/validation_campaign/s20/summary-20260923.md) |
| `S21` | Image matching and partial-import confirmation | `S20`, `S15` | `PASS` | Full/partial/no match, confirmation before import, rendered counts, required-column errors, case-insensitive stems, and persisted source rows passed in service, API, and rendered browser checks. [2026-09-23 S21 summary](../QA/validation_campaign/s21/summary-20260923.md) |
| `S22` | Dataset processing and integrity | `S21` | `PASS` | Minimum-one sampling, real tokenizer processing, 50%/100% sample counts, 3/1 and 6/2 train/validation splits, same-name identity with two run-history records, missing-image failure, and restart metadata passed on the isolated eight-row fixture. Functional checks only; no scale/performance claim. [2026-09-23 S22 summary](../QA/validation_campaign/s22/summary-20260923.md) |
| `S23` | Dataset viewer and deletion | `S22` | `PASS` | Rendered long-path row containment and eight-image viewer navigation passed in the original S23 browser run. The 2026-09-24 API regression now returns 409 and names dependent processed datasets while preserving source and training rows; the sample remains loadable, and processed-then-source deletion completes with no foreign-key violations and SQLite integrity `ok`. The 2026-09-28 current-head browser regression recaptured the rejected-delete message, showed the dependent name in the rendered Dataset error state, deleted the processed row from Training first, and then removed the source row from Dataset. See [S23 browser deletion summary](../QA/validation_campaign/s23-delete-ui-20260928/summary-20260928.md), [S23 follow-up](../QA/validation_campaign/s23/summary-20260924.md), [original UI observations](../QA/validation_campaign/s23-s26/s23-ui-observations.json), and [original deletion receipt](../QA/validation_campaign/s23-s26/s23-deletion.json). |
| `S24` | Minimal real training | `S22` | `PASS` | One real synthetic-dataset epoch completed on CUDA with a ready checkpoint, persisted configuration, provenance, and epoch history. The exact pinned BEiT revision and weight hash are recorded. This is software-path evidence only; no quality or performance claim. [Training receipt](../QA/validation_campaign/s23-s26/s24-s25-training.json); [checkpoint API evidence](../QA/validation_campaign/s23-s26/s26-checkpoint-api.json). |
| `S25` | Training stop and resume | `S24` | `PASS` | A separate 100-epoch CUDA job cancelled at 5%/epoch 5 and reached terminal cancellation with worker exit. The S24 checkpoint then resumed for one epoch; persisted history advanced from one to two epochs. [Training receipt](../QA/validation_campaign/s23-s26/s24-s25-training.json); [checkpoint API evidence](../QA/validation_campaign/s23-s26/s26-checkpoint-api.json). |
| `S26` | Checkpoint registry, metadata, and deletion | `S24` | `PASS` | Ready listing and metadata contained dataset/configuration/history; deletion of a referenced checkpoint returned 409 and retained artifact/registry; unsafe path returned 400; unreferenced deletion removed both artifact and registry. `ISSUE-003` was traced to incomplete test fixture history and corrected; isolated list and database had no stale `e2e_delete_*` registrations or matching warnings. [Checkpoint API evidence](../QA/validation_campaign/s23-s26/s26-checkpoint-api.json); [ISSUE-003 trace](../QA/validation_campaign/s23-s26/issue-003.json); [summary](../QA/validation_campaign/s23-s26/summary-20260923.md). |
| `S27` | Successful dataset validation | `S22` | `PASS` | Full-dataset validation on an eight-row synthetic fixture completed twice with text, image, and pixel-distribution metrics. The current-head workload recheck also completed all three metrics and retrieved the persisted report with the expected `104` total words, `20` unique words, and `64x64` image dimensions. Both jobs succeeded; the saved report returned HTTP 200 with matching dataset, sample size, metrics, and persisted timestamp; the rebuilt route rendered the report successfully. [S27 summary](../QA/validation_campaign/s27/summary-20260923.md); [fixture manifest](../QA/validation_campaign/s27/fixture-manifest.json); [current-head runtime receipt](../QA/validation_campaign/s51-s53-20260928/current-head-runtime-contention-receipt.json); [API receipts and backend log](../QA/validation_campaign/s27/). Synthetic technical workflow only; no clinical, representative-data, or scale claim. The backend returned an empty `artifacts` map, so separate artifact-file generation was not validated. The browser capture was reviewed inline but not exported as a standalone PNG. |
| `S28` | Successful checkpoint evaluation | `S24`, `S27` | `PASS` | On 2026-09-25 the exact pinned BEiT revision `f02e8f77db4703e3fbd3766e3375a4619c5a4863` and verified weight hash were restored in an isolated resource root. The current-head recheck also evaluated the retained checkpoint against the S27 fixture and retrieved the persisted report with `loss=2.0254554748535156` and `accuracy=0.8571428656578064`. The S27 eight-row fixture was processed, one CUDA epoch created checkpoint `XREPORT_20260925T141533`, and the UI/API evaluation job completed with a persisted report. The report remained visible after closing/reopening and browser reload. Synthetic technical workflow only; no clinical, representative-data, scale, or release-performance claim. [S28/S34/S53 summary](../QA/validation_campaign/s28-release-20260925/summary.md); [current-head runtime receipt](../QA/validation_campaign/s51-s53-20260928/current-head-runtime-contention-receipt.json); [workflow receipts](../QA/validation_campaign/s28-release-20260925/workflow-api-receipts.json); [encoder receipt](../QA/validation_campaign/s28/encoder-download-20260925.json). |

### Tier 3 — Inference

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S30` | Inference catalogue and public/custom separation | `S26` | `PASS` | The 2026-09-24 evidence is historical five-model catalogue coverage. The current 2026-09-29 catalogue has four public entries after CXRMate-ED retirement; the manifest now requires unique refs and valid entries without a fixed cardinality invariant. Unit/API coverage verifies public/custom identity and retired-ref rejection. [Historical campaign summary](../QA/validation_campaign/s30-s32-20260924/summary.md); [retirement receipt](../QA/validation_campaign/s35-s36-20260926/cxrmate-ed-retirement-diagnostic-20260929.json); [UI observation](../QA/validation_campaign/s35-s36-20260926/cxrmate-ed-retirement-ui-observation-20260929.json); [API E2E](../../app/tests/e2e/test_inference_api.py). |
| `S31` | Lightweight public-model installation lifecycle | `S30` | `PASS` | Revalidated on 2026-09-24 with an isolated exact-pinned CXRMate Multi install: remote install/staging, one-file corruption and repair, real inference readiness promotion, stop/restart with unchanged snapshot and no redownload, deletion, and persistent inference history passed. The offline load check did not establish that the separate canonical resource matches the pinned manifest: the current comparison found one differing `modelling_multi.py`; see `ISSUE-006`. PASS remains limited to the fresh isolated install and one public image, as technical lifecycle evidence only. [S31 summary](../QA/validation_campaign/s31/summary-20260924.md); [current canonical comparison](../QA/validation_campaign/s33-s52-20260924/canonical-model-recheck.json); [current S28/S35/S37 recheck](../QA/validation_campaign/s33-s52-20260924/gate-recheck.json). |
| `S32` | Inference input and browser-state validation | `S30` | `PASS` | Rechecked 2026-09-24: six live API E2E tests cover model context, all profiles, current-image count, image type/emptiness, invalid profile, and aggregate 64 MiB + 1 byte rejection (HTTP 413). Rendered UI confirmed model-dependent context controls, profile changes, disabled generate state, and no draft. No user image was selected in the UI and no generation job was started; API payloads were synthetic. [Campaign summary](../QA/validation_campaign/s30-s32-20260924/summary.md); [API log](../QA/validation_campaign/s30-s32-20260924/api-e2e-final.log); [browser observations](../QA/validation_campaign/s30-s32-20260924/browser-observations.json); [CXRMate-ED screenshot](../QA/validation_campaign/s30-s32-20260924/cxrmate-ed-status-20260924.jpg). |
| `S33` | Complete browser inference workflow | `S31`, `S32` | `PASS` | On 2026-09-24, the rendered inference route uploaded the public fixture (SHA-256 `4570a9524d57cb2697ea577ac5d36553cf28e0d190d347edda1fe12f07d54751`) and ran exact revision `330721b9aa5bba201a3eb88eba4dd9a6607f3e7a` in the isolated runtime. Browser-observed success, cancel, retry, persisted report history, saved edit/reload with original retained, provenance, copy, and exported Findings/Impression content passed. Export file exists and its contents/hash were checked, although the browser download event was not surfaced by the adapter. One model and one public image only; catalogue `validation_status=pending`; no clinical or quality claim. [S33/S52 summary](../QA/validation_campaign/s33-s52-20260924/summary.md); [UI observations](../QA/validation_campaign/s33-s52-20260924/ui-observations.json); [client unit logs](../QA/validation_campaign/s33-s52-20260924/client-unit-inference.log); [production build log](../QA/validation_campaign/s33-s52-20260924/client-production-build.log). |
| `S34` | Custom XREPORT checkpoint inference | `S24`, `S26` | `PASS` | The generated S28 checkpoint was listed as a ready custom model, selected in the rendered inference catalogue, and used for one synthetic multipart inference. The persisted history and expanded provenance survived browser reload and identified `xreport:XREPORT_20260925T141533`, `keras_checkpoint`, `fixed_224`, and `xreport_beit`. Native file-picker authorization was unavailable to the browser adapter, so the fixture entered through the local multipart route. Technical synthetic scope only; no report-quality claim. [S28/S34/S53 summary](../QA/validation_campaign/s28-release-20260925/summary.md); [browser observations](../QA/validation_campaign/s28-release-20260925/browser-workflow-observations.json); [workflow receipts](../QA/validation_campaign/s28-release-20260925/workflow-api-receipts.json). |
| `S35` | CXRMate-ED qualification and retirement | `S31` | `RESOLVED` / retired | The bounded rescue attempt replayed the pinned revision `68251c7605067ddbea330413aade032713fd2192` with the published `max_length=256`, `num_beams=4` contract against the three approved fixtures. All cases completed, but one identical full Findings/Impression report was produced. The separate reference process was blocked by the unavailable pinned runtime and a missing transitive dynamic-module file, so no corrected quality PASS is claimed. CXRMate-ED was removed from the active catalogue and adapter set; history, provenance, and the downloaded snapshot remain preserved. [Retirement receipt](../QA/validation_campaign/s35-s36-20260926/cxrmate-ed-retirement-diagnostic-20260929.json); [summary](../QA/validation_campaign/s35-s36-20260926/cxrmate-ed-retirement-summary-20260929.md); [backend regression](../../app/tests/unit/test_inference_service.py); [UI/API regressions](../../app/client/src/app/pages/job-cancellation.pages.spec.ts). |

### Tier 4 — Heavy Providers and Desktop

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S36A` | CheXOne technical and broader quality gate | `S33` | `PASS` | Installed and verified exact revision `0c350e6852ea08f9d9baf3b7595c1a10d4849927` in the isolated root, then ran the three manifest cases independently and sequentially. All cases produced non-empty Findings-only output with correct provenance, no timeout/OOM/runtime error, measured CUDA/resource use, and a successful reload/reuse sentinel (`4` loads including the repeat). Conservative manual review passed for case responsiveness, section semantics, prompt/reasoning leakage, pathological repetition, and unresolved collapse. The current disposable rendered catalogue also showed `validation policy=passed`, `validation evidence=passed`, and Findings-only output. This is research-only technical evidence, not diagnostic-accuracy or release-readiness evidence. [S36A receipt](../QA/validation_campaign/s35-s36-20260926/s36a-chexone-aggregate-technical-receipt.json); [manual review](../QA/validation_campaign/s35-s36-20260926/manual-review.json); [rendered catalogue observation](../QA/validation_campaign/s35-s36-20260926/browser-catalogue-observation.json). |
| `S36B` | CXRMate-2 technical and broader quality gate | `S33` | `PASS` | Installed and verified exact revision `aa8e2d16470e20671acf049687b4707c9bf2f2b5` in the isolated root, then ran the same three manifest cases independently and sequentially. All cases produced non-empty Findings and Impression output with correct provenance, no timeout/OOM/runtime error, measured very-high CUDA/resource use, and a successful reload/reuse sentinel (`4` loads including the repeat). Conservative manual review passed for case responsiveness, section semantics, prompt/reasoning leakage, pathological repetition, and unresolved collapse. The current disposable rendered catalogue also showed `validation policy=passed`, `validation evidence=passed`, and Findings/Impression output. This is research-only technical evidence, not diagnostic-accuracy or release-readiness evidence. [S36B receipt](../QA/validation_campaign/s35-s36-20260926/s36b-cxrmate2-aggregate-technical-receipt.json); [manual review](../QA/validation_campaign/s35-s36-20260926/manual-review.json); [rendered catalogue observation](../QA/validation_campaign/s35-s36-20260926/browser-catalogue-observation.json). |
| `S37` | MedGemma gated-provider behavior | `S30` | `PASS` | The gated-provider behavior contract passed without model access: the live catalogue reports `access_policy=gated`, `not_installed`, the terms message, and the access URL; `generate_reports` rejects a gated non-ready public model before job start (HTTP 409); the rendered UI shows `NOT INSTALLED`, the access notice/link, and a disabled `Access required` Generate button. `ISSUE-002` (authorized gated access/credential for a real install) remains open. [2026-10-01 S37 summary](../QA/validation_campaign/s37/summary-20261001.md); [live receipt](../QA/validation_campaign/s11-s37-20261001/s37-s11-live-receipt.json); [UI receipt](../QA/validation_campaign/s11-s37-20261001/s37-gated-ui-receipt.json) |
| `S40` | Tauri development shell | `S02`, `S03` | `PASS` | Current native matrix is `10 PASS`, `0 FAIL`, `0 UNTESTED`: startup, six routes, refresh/back/forward, settings save/restart, keyboard/modal/focus, backend stop/error, backend retry/recovery, second-instance policy, close/cleanup, and controlled delayed readiness passed in the CPU Tauri development shell. S40-06 showed `XREPORT is still initializing`, started the backend after 18 seconds, and reached the ready `Inference` workspace in the same native window. This is source CPU-shell evidence only; it does not close packaged, genuine no-GPU, or S51 boundaries. [S40 reconciliation](../QA/validation_campaign/s40-s42-s51-release-20260930/s40-reconciliation-20260930.md); [S40-06 note](../QA/validation_campaign/s40-s42-s51-release-20260930/s40-06-delayed-readiness-20261001.md); [S40-06 receipt](../QA/desktop/s40-06-delayed-readiness-20261001.json); [native follow-up](../QA/validation_campaign/s40-s42-s51-release-20260930/native-follow-up-20260930.md); [historical nine-scenario receipt](../QA/validation_campaign/s40-s42-s51-release-20260930/native-recheck-20260930/native-final-validation.json); [release summary](../QA/validation_campaign/s40-s42-s51-release-20260930/summary-20260930.md). |
| `S41` | CPU packaged desktop | `S40` | `PASS` | The current v3.1.0 CPU portable/MSI pair was rebuilt from source commit `301a5a510e494c501d6c25eb22d5baea824e936a` with `dirty_tree=false` and passed the official artifact verifier; packaged startup/readiness/health/frontend serving, graceful close, backend/listener removal, contract cleanup, and retained native CPU interactions passed. The full MSI install → registered → launch → graceful close → uninstall lifecycle also passed on one elevated transaction (install/uninstall `msiexec` exit `0`, contract files removed, registration and `C:\Program Files\XREPORT CPU` removed). [current release record](../QA/validation_campaign/s40-s42-s51-release-20260930/summary-20260930.md); [packaged native validation](../QA/validation_campaign/s40-s42-s51-release-20260930/packaged-native-validation-20261001.md); [MSI lifecycle receipt](../QA/validation_campaign/msi-lifecycle-20261001/cpu-receipt-20261001.json); [runtime audit](../QA/desktop/runtime-cpu-3.1.0.json); [verification](../QA/desktop/verification-cpu-3.1.0.json); [smoke receipt](../QA/desktop/smoke-cpu-3.1.0.json). |
| `S42` | CUDA packaged desktop | `S41` or compatible hardware | `PARTIAL` | The current v3.1.0 CUDA portable/MSI pair was rebuilt from source commit `301a5a510e494c501d6c25eb22d5baea824e936a` with `dirty_tree=false` and passed the official artifact verifier. The current packaged CUDA inference completed with `resolved_device=cuda:0`, `cuda_available=true`, and `cuda_used=true`, and persisted the same provenance in history; the retained native receipt remains historical relative to the current artifact directory. The full CUDA MSI install → registered → launch (health `ok`, variant `cuda`) → graceful close → uninstall lifecycle passed on one elevated transaction, closing the MSI installation/uninstallation boundary for CUDA. A genuine no-GPU package lane remains unrun. The supported repair workflow restored all eight canonical CXRMate Multi files to the S31 manifest, and one disposable Generate call recorded verified integrity; `ISSUE-006` is resolved for the current local resource. See the [current packaged CUDA inference](../QA/validation_campaign/s40-s42-s51-release-20260930/packaged-cuda-inference-current-20261001.md), [MSI lifecycle receipt](../QA/validation_campaign/msi-lifecycle-20261001/cuda-receipt-20261001.json), [canonical repair note](../QA/validation_campaign/s40-s42-s51-release-20260930/canonical-model-repair-20261001.md), [2026-10-01 release-validation record](../QA/validation_campaign/s40-s42-s51-release-20260930/summary-20260930.md), [packaged native validation](../QA/validation_campaign/s40-s42-s51-release-20260930/packaged-native-validation-20261001.md), [historical packaged inference note](../QA/validation_campaign/s40-s42-s51-release-20260930/packaged-cuda-inference-clean-20261001.md), [runtime audit](../QA/desktop/runtime-cuda-3.1.0.json), [artifact verification](../QA/desktop/verification-cuda-3.1.0.json), and [packaged smoke](../QA/desktop/smoke-cuda-3.1.0.json). |
| `S43` | Windows launcher maintenance actions | `S13` | `PASS` | The current launcher functions passed disposable data-root/cache/checkpoint/database/model/log cleanup and idempotent repeats, source FastAPI/Node tree termination, and listener cleanup. CPU and CUDA portable packages reached readiness with recorded Tauri/backend parent-child ownership; official `KillProcesses` removed each packaged root, backend child, and dynamic listener. Confirmed `RemoveDesktopRelease` removed release packages, staging, generated runtime/UI, and target outputs; its repeat was an idempotent no-op and canonical `data/` fingerprints were unchanged. S43 is limited to launcher maintenance; packaged native WebView, no-GPU, and performance boundaries remain outside this PASS. [S43 packaged-maintenance summary](../QA/validation_campaign/s43-packaged-maintenance-20260928/summary-20260928.md); [S43 validation receipt](../QA/validation_campaign/s43-packaged-maintenance-20260928/validation-receipt.json); [cleanup transcript](../QA/validation_campaign/s43-packaged-maintenance-20260928/remove-desktop-release.log); [selector regression](../QA/validation_campaign/s43-packaged-maintenance-20260928/process-selector-harness.ps1); [current data-root summary](../QA/validation_campaign/s01-s43-s50-data-root-20260928/summary-20260928.md); [current maintenance harness receipt](../QA/validation_campaign/s01-s43-s50-data-root-20260928/launcher-data-root-harness-receipt.json); [shutdown safety receipt](../QA/validation_campaign/s01-s43-s50-data-root-20260928/final-safety-receipt.json). |

### Tier 5 — Resilience and Baseline

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S50` | Restart and corruption recovery | Tiers 1–3 | `PASS` | On current revision `7bf8402`, the official source launcher persisted Settings seed `20260928` across an ordinary stop/relaunch and rendered it in Settings. Injected `datasets.s50_unexpected` schema drift kept SQLite integrity readable but caused backend startup to fail closed; the frontend stayed available, rendered the unavailable alert and Retry state, and the failed start left the database unchanged. A byte-for-byte backup restore recovered the original schema/seed; post-restore and repeated clean-start health, API readback, and integrity passed. The synthetic report persistence/readback remains evidenced by the earlier 2026-09-28 S50 receipt and was not repeated in this root recheck. [current data-root summary](../QA/validation_campaign/s01-s43-s50-data-root-20260928/summary-20260928.md); [validation receipt](../QA/validation_campaign/s01-s43-s50-data-root-20260928/validation-receipt.json); [drift receipt](../QA/validation_campaign/s01-s43-s50-data-root-20260928/drift-injection-receipt.json); [backup restore receipt](../QA/validation_campaign/s01-s43-s50-data-root-20260928/backup-restore-receipt.json); [clean restart receipt](../QA/validation_campaign/s01-s43-s50-data-root-20260928/clean-restart-receipt.json); [earlier S50/S52 summary](../QA/validation_campaign/s50-s52-20260928/summary-20260928.md); [earlier structured receipt](../QA/validation_campaign/s50-s52-20260928/s50-s52-validation-receipt.json). |
| `S51` | Concurrent operations and resource contention | Tiers 2–3 | `PARTIAL` | The maintained source-runtime harness adds explicit CUDA/CPU/unavailable-GPU lanes, scale-aware fixture and checkpoint metadata, clean/warm run metadata, expected terminal-status assertions, worker PID/phase/latency diagnostics, per-worker process attribution where measurable, nonterminal-job detection, bounded stranded-job cleanup, and database/resource reconciliation. Three independent clean-restart CUDA matrices passed. The 2026-10-01 dirty-tree follow-up additionally passed the eight-scenario CPU scale-1 matrix, the real 64-row scale-8 CPU matrix, and a software-masked unavailable-GPU fallback matrix with cleanup and SQLite integrity checks clean. The packaged CPU receipt also observed `/api/jobs` in `running` state immediately before native close, removed the package/backend/listener/contracts, and reopened with zero running jobs. Genuine GPU-less hardware, hosted-CI, representative-data, and original slow-initialization attribution remain open; native Training-form submission and graceful user-cancellation semantics remain outside the packaged API-started receipt. [native follow-up](../QA/validation_campaign/s40-s42-s51-release-20260930/native-follow-up-20260930.md); [packaged native validation](../QA/validation_campaign/s40-s42-s51-release-20260930/packaged-native-validation-20261001.md); [packaged active-training note](../QA/validation_campaign/s40-s42-s51-release-20260930/packaged-active-training-close-reopen-20261001.md); [packaged active-training receipt](../QA/validation_campaign/s40-s42-s51-release-20260930/packaged-active-training-close-reopen-20261001.json); [harness](../../app/scripts/validate_resilience_baseline.py); [harness tests](../../app/tests/unit/test_validate_resilience_baseline.py); [follow-up summary](../QA/validation_campaign/s51-s53-20260930/s51-followup-summary-20260930.md); [CPU receipt](../QA/validation_campaign/s51-s53-20260930/s51-cpu-source-run-20261001.json); [scale-8 receipt](../QA/validation_campaign/s51-s53-20260930/s51-scale8-source-cpu-20261001.json); [emulated fallback receipt](../QA/validation_campaign/s51-s53-20260930/s51-unavailable-gpu-emulated-20261001.json); [prior consolidated receipt](../QA/validation_campaign/s51-s53-20260930/s51-followup-receipt-20260930.json). |
| `S52` | Accessibility and responsive workflow audit | Stable Tiers 2–3 UI | `PASS` | The current rebuilt source client passed the focused S52 browser suite (`2 passed, 14 deselected`) across Inference, Reports, Dataset, Training, and Settings at `320x640`, `390x844`, `640x800`, `1024x720`, and `1440x900`, with 25 dimension-verified screenshots. The run additionally checked route semantics, accessible names/labels, ARIA references, keyboard-only modal/tour behavior and focus restoration, settings tab arrow navigation, training disclosure keys, reduced motion, responsive reflow, and bounded model-catalogue scrolling. A real Narrator/Speech Recap output check was not performed and remains non-gating. The canonical protected dist could not be overwritten by the launcher (`EPERM`); the same current source built successfully into an isolated writable output and was the browser-tested bundle, so this PASS is source/browser scoped. [2026-09-29 receipt and screenshots](../QA/validation_campaign/s51-s53-20260929/closure-summary-20260929.md); [structured receipt](../QA/validation_campaign/s51-s53-20260929/s52-browser-receipt-20260929.json); [committed E2E coverage](../../app/tests/e2e/test_angular_ui.py); [settings unit coverage](../../app/client/src/app/pages/settings.page.spec.ts). |
| `S53` | Performance and long-operation baseline | Stable representative workflows | `PASS` | Final source revision `a10de59` imported and processed the deterministic scale-8, 64-row fixture through the real S20/S21/S22 path, then completed the full ten-scenario matrix twice: three matched one-epoch CUDA runs, three matched one-epoch CPU runs, one three-epoch CUDA run, and three inference repeats. The second run followed a clean backend restart with the exact tokenizer forced offline. Every scenario completed without execution errors; raw receipts preserve submission/wall time, CPU, process working set, system memory, GPU utilization/memory, terminal state, warnings, and no-threshold median/range summaries. XREPORT history now persists requested/resolved device and CUDA availability/use fields for every inference repeat. Cleanup left no running jobs, retained only the baseline checkpoint and 64-row processed dataset, and SQLite integrity was `ok`. This PASS is source-runtime technical baseline evidence only; packaged timing, native WebView, no-GPU desktop performance, and clinical-quality evidence remain outside S53. [2026-09-30 summary](../QA/validation_campaign/s51-s53-20260930/summary-20260930.md); [scale fixture](../QA/validation_campaign/s51-s53-20260930/s53-scale8-fixture/fixture-manifest.json); [first final-revision receipt](../QA/validation_campaign/s51-s53-20260930/s53-scale8-run3-after-provenance.json); [clean-restart receipt](../QA/validation_campaign/s51-s53-20260930/s53-scale8-run4-after-restart-provenance.json). |

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
