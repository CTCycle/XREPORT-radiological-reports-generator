# XREPORT Validation Campaign Ledger

Last updated: 2026-09-22

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

- Code revision at campaign start: `481605b1b87035b8deb03edaefdbfc090f8f1b23` (`develop`). The 2026-09-22 implementation was tested in a worktree based on that revision before commit; see the dated execution summary.
- Remote CI run `35598132452` is red on this revision: backend unit tests recorded
  `131 passed, 1 failed` in
  `test_concurrent_service_initialization_keeps_ml_imports_lazy`; later CI
  stages were skipped. The last verified fully green baseline is run
  `35348243935` at `5d469126385f1f414023d91845a8c8a317702d3f`, which predates
  the current model-import and startup-gate changes.
- Windows reproduced the named test successfully once and for ten consecutive
  repetitions, and the complete backend unit suite passed (`132 passed`). This
  does not clear S00 because the current remote CI result remains unresolved.
- On 2026-09-22, the source launcher, built frontend server, rendered startup
  gate, and local static/client/backend gates passed within the recorded
  Windows scope. S01–S03 evidence is refreshed in
  [the 2026-09-22 execution summary](../QA/validation_campaign/tier-0/summary-20260922.md).
  S00 remains **NOT CLEARED**: the last observed remote run is still the red
  baseline above, and no current-change remote CI result is available here.

## Tier 0 Slice Ledger — Current Execution

| Slice | Capability | Exists | Exercised | Status | Scenarios and regressions | Evidence | Remaining gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `S00` | Restore current CI baseline | yes | yes | `FAIL` | Named lazy-import test passed 10/10 on Windows; full backend unit suite passed. The last observed remote CI failed 131/1 on the starting revision; no current-change CI result is available. | [Tier 0 execution summary](../QA/validation_campaign/tier-0/summary-20260921.md); [CI run 35598132452](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/35598132452) | Obtain the full remote traceback, classify the platform/state defect, apply only the narrow owner fix if needed, rerun the test at least 10 times, then rerun every skipped CI stage. |
| `S01` | Source backend startup and SQLite readiness | yes | yes | `PASS` | The normal Windows launcher started FastAPI and the built UI; the in-browser readiness gate reached ready state and loaded the catalogue. Full local Python run passed 149 tests, including unchanged migration/schema-drift checks. | [2026-09-22 execution summary](../QA/validation_campaign/tier-0/summary-20260922.md); [2026-09-21 execution summary](../QA/validation_campaign/tier-0/summary-20260921.md) | Complete the still-open launcher port-race/failure and build dependency-invalidation scenarios before broadening the source-startup claim. |
| `S02` | Built frontend startup gate and backend recovery | yes | yes | `PASS` | `run_tests.bat` and focused startup-gate E2E passed; the in-app browser observed the launch shell transition to the ready inference UI. Nine built-server unit tests passed for static files, SPA fallback, API proxy/body/status, 502 recovery, traversal, and symlink protection. | [2026-09-22 execution summary](../QA/validation_campaign/tier-0/summary-20260922.md); [startup screenshot](../QA/xreport-startup-gate.png) | Explicit slow/unavailable/retry timing and post-ready ordinary feature-error evidence remain open. |
| `S03` | Static quality, API contract, and client build gates | yes | yes | `PASS` | Ruff and Pyright passed; portable Node 22.22.3 production build and explicit launcher rebuild passed; client lint and 29 unit tests passed; the full Python suite passed 149/1 skipped. | [2026-09-22 execution summary](../QA/validation_campaign/tier-0/summary-20260922.md) | `npm ci` and remote current-head CI were not run for this change set; S00 remains open until current-head CI is green. |

### Tier 0 Evidence Notes

The 2026-09-21 launcher attempt to open the URL through Windows `Start-Process`
returned access denied in the managed environment. On 2026-09-22 the corrected
launcher completed successfully; the same local URL was opened in the Codex
in-app browser and rendered the ready workspace. Its launch-owned process trees
were terminated by verified roots and ports `5003` and `8003` were confirmed
clear. Port-guard cases not exercised are enumerated in the dated summary.

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
| `S10` | Routing, navigation, theme, and guidance persistence | `S02`, `S03` | `PARTIAL` | Direct-load/refresh every route, active navigation, all theme modes, persisted theme/guidance state, and console cleanliness. |
| `S11` | Runtime settings persistence, limits, reset, and hidden-field protection | `S01` | `VALIDATED` for current exercised surface | GET/PATCH/reset payloads, database row before/after, restart persistence, bounds, and captured-value behavior for active jobs. |
| `S12` | Generic job lifecycle | `S01` | `PARTIAL` | Deterministic start/list/filter/poll/complete, unknown ID, cancellation, transient poll failure, and typed recoverable failure timeline. |
| `S13` | SQLite migration and restart persistence | `S01` | `PARTIAL` | Settings and representative data before/after restart, current head, incompatible schema, interrupted migration, foreign-key behavior. |
| `S14` | PostgreSQL persistence contract | `S00` | `UNTESTED` | PostgreSQL 16 migration, transactions, restart, advisory locking, and sanitized connection failure. |
| `S15` | Local filesystem access and path controls | `S11` | `PARTIAL` | Enable/disable browse, permitted and invalid paths, empty folder, image selection, and UI recovery. |

### Tier 2 — Dataset and Train/Evaluate Backbone

| Slice | Capability | Prerequisite | Current status | Required evidence |
| --- | --- | --- | --- | --- |
| `S20` | CSV/XLSX upload parsing and explicit upload identity | `S10` | `PARTIAL` | Valid/invalid formats, delimiter, empty/corrupt input, 16 MiB boundary, two independent uploads, and invalid ID. |
| `S21` | Image matching and partial-import confirmation | `S20`, `S15` | `PARTIAL` | Full/partial/no match, confirmation before import, visible counts, required-column errors, case handling, and persisted source rows. |
| `S22` | Dataset processing and integrity | `S21` | `PARTIAL` | Sampling, minimum-one retention, validation fractions, tokenizer choices, naming conflicts, missing image failure, restart metadata. |
| `S23` | Dataset viewer and deletion | `S22` | `UNTESTED` | Rendered image viewer, deletion consequences, source/processed state, and relational cleanup. |
| `S24` | Minimal real training | `S22` | `UNTESTED` | Real small CPU/CUDA-appropriate training receipt, checkpoint artifact, provenance, and cleanup. |
| `S25` | Training stop and resume | `S24` | `PARTIAL` | Managed-process termination, persisted progress, resume from a real checkpoint, and no orphan process. |
| `S26` | Checkpoint registry, metadata, and deletion | `S24` | `PARTIAL` | Complete-artifact registration, listing, metadata, safe deletion, and stale-registration cleanup. |
| `S27` | Successful dataset validation | `S22` | `UNTESTED` | Non-empty validation completion, metric/report persistence, and rendered report review. |
| `S28` | Successful checkpoint evaluation | `S24`, `S27` | `UNTESTED` | Compatible real checkpoint evaluation, dataset resolution, metrics, and persisted report retrieval. |

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
