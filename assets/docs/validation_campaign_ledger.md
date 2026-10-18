# XREPORT Validation Campaign Ledger

Last updated: 2026-10-18

This is the slice ledger for the long-term XREPORT validation campaign.
[`project_status_ledger.md`](project_status_ledger.md) remains the canonical
aggregate status catalog; this document records slice order, prerequisites,
evidence scope, and campaign progress. The campaign is technical/software
validation for research-use workflows. It does not establish clinical validity,
diagnostic performance, or regulatory approval.

## Campaign Rules

- A feature existing, having an automated test, being exercised, and passing
  when observed are four separate facts. Keep them separate in every claim.
- Every result is tied to an exact revision, environment, fixture, scenario set,
  and durable record. Slice history is summarized in this ledger; bulky
  captures are archived through Git history rather than kept in the working
  tree (see Evidence Policy).
- `PASS` is scoped to the stated slice. A technical model receipt is not a
  quality study, and a route-render test is not a complete workflow proof.
- A red hard gate stops downstream claims. Do not spend campaign effort on
  dependent workflows while the current baseline is red.
- Use the sequence **inspect → execute → observe → diagnose → surgically fix →
  retest → record**. Capture the original failure before changing code and run
  the adjacent regression set after a fix.
- Keep disposable databases, model/tool caches, pytest state, and generated
  bundles under `runtimes/cache`.
- Public-model evidence must identify the exact model revision, fixture
  provenance and de-identification statement, requested/actual device, output
  contract, timing, memory, and warning state.
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

## Slice Ledger

Status values are scoped to the recorded evidence. Dates name the most recent
recorded run for the row. Detailed per-run captures that were once tracked are
archived in Git history; see Evidence Policy below.

### Tier 0 — Foundational Health

| Slice | Capability | Status | Evidence scope and remaining gap |
| --- | --- | --- | --- |
| `S00` | Restore current CI baseline | `PASS` | Hosted CI run `36422997859` passed every configured gate on validation revision `7bf8402f90bdbf2181f13190782b67ba51d445d9`; the later hosted run `36586991512` passed for `064278f8`. Earlier green and failed runs remain historical. Re-run when source or test behavior changes before a release. |
| `S01` | Source backend startup and SQLite readiness | `PASS` (scoped) | The official launcher initialized a fresh isolated process-override root, reached backend and frontend-proxy health `200`, rendered readiness, and initialized SQLite at Alembic head `e91a4f6c2d73`; a saved seed survived stop/relaunch, and the default/override data-root contract was rechecked. Aggregate source startup remains `PARTIAL`: interactive termination/ownership races, output deletion, rebuild-after-source-change, package-lock/npm-ci recovery, and edit-during-build are untested. |
| `S02` | Built frontend startup gate and backend recovery | `PASS` | Controlled-clock health responses rendered slow and unavailable states, exposed retry, recovered to ready, suppressed routes before readiness, and kept a synthetic post-ready catalogue 503 inline. Timing is controlled-clock E2E evidence, not a wall-clock outage; packaged CPU/CUDA startup stays outside. |
| `S03` | Static quality, API contract, and client build gates | `PASS` | Hosted runs cover the backend, client, and PostgreSQL gates on the validation revision; an adjacent local recheck passed `166` backend unit tests, Ruff, Pyright (`0` errors/warnings), Angular production build, client lint, and `43` client unit tests. The Angular production builder later crashed intermittently with Windows `0xC0000005` locally, so no new full local production-build PASS is claimed beyond hosted CI. |

### Tier 1 — State and Infrastructure

| Slice | Capability | Status | Evidence scope and remaining gap |
| --- | --- | --- | --- |
| `S10` | Routing, navigation, theme, and guidance persistence | `PASS` | 2026-09-22 browser run: direct-load/refresh for every route, active navigation including parameterized child routes, all theme modes, persisted theme/guidance state, manual guidance replay, and clean console/request collectors. |
| `S11` | Runtime settings persistence, limits, reset, and hidden-field protection | `PASS` | 2026-10-01 live GET/PATCH/reset on a disposable-root SQLite backend: public projection, persisted seed readback, bounds (`polling_interval` `0.25`–`60.0`), hidden-field protection (`inference.device` 422), null rejection, and reset-to-defaults; backend unit and settings-page client specs also passed. |
| `S12` | Generic job lifecycle | `PASS` | 2026-09-22 deterministic API coverage: start/list/type-and-status filtering/poll/complete, unknown ID, active cancellation through terminal state, and typed recoverable failure; client tests cover transient/repeated transport failures and missing-job termination. |
| `S13` | SQLite migration and restart persistence | `PASS` | 2026-09-23: fresh isolated SQLite reached head `e91a4f6c2d73`; a non-default setting and synthetic inference history with a linked report survived stop/relaunch and API readback; unversioned/unknown schema rejection, migration rollback, schema drift, FK enforcement, orphan rejection, and child cascade were verified. |
| `S14` | PostgreSQL persistence contract | `PASS` | 2026-09-23 hosted PostgreSQL 16: migrations to head `e91a4f6c2d73`, schema contract, settings persistence across reinitialization, concurrent first-time initialization under advisory locks, and a refused connection that did not expose credentials. |
| `S15` | Local filesystem access and path controls | `PASS` | 2026-09-23 API and rendered scenarios: enabled/disabled access, accessible image folder, invalid/non-folder paths, empty folder, valid image selection with count, and recovery after an invalid path. Filesystem access/selection only; upload, matching, and processing are covered by S20–S23. |

### Tier 2 — Dataset and Train/Evaluate Backbone

| Slice | Capability | Status | Evidence scope and remaining gap |
| --- | --- | --- | --- |
| `S20` | CSV/XLSX upload parsing and explicit upload identity | `PASS` | 2026-09-23 API and service checks: comma/semicolon CSV, generated XLSX, empty/corrupt/unsupported files, exact and over-limit payloads, independent upload IDs/content, and unknown upload ID. |
| `S21` | Image matching and partial-import confirmation | `PASS` | 2026-09-23 service/API/browser: full/partial/no match, confirmation before import, rendered counts, required-column errors, case-insensitive stems, and persisted source rows. |
| `S22` | Dataset processing and integrity | `PASS` | 2026-09-23 isolated eight-row fixture: minimum-one sampling, real tokenizer processing, 50%/100% sample counts, 3/1 and 6/2 train/validation splits, same-name identity with two run-history records, missing-image failure, and restart metadata. Functional checks only; no scale claim. |
| `S23` | Dataset viewer and deletion | `PASS` | Rendered long-path row containment and eight-image viewer navigation; dependent source deletion returns HTTP 409 with dependent names; the 2026-09-28 browser flow recaptured the rejected delete, processed-first deletion, then source removal with SQLite integrity `ok`. |
| `S24` | Minimal real training | `PASS` | One real synthetic-dataset epoch completed on CUDA with a ready checkpoint, persisted configuration, provenance, and epoch history at the pinned BEiT revision. Software-path evidence only; no quality or performance claim. |
| `S25` | Training stop and resume | `PASS` | A separate 100-epoch CUDA job cancelled at 5% / epoch 5 reached terminal cancellation with worker exit; the S24 checkpoint resumed for one epoch and persisted history advanced from one to two epochs. |
| `S26` | Checkpoint registry, metadata, and deletion | `PASS` | Ready listing and metadata with dataset/configuration/history; deletion of a referenced checkpoint returned 409; unsafe path returned 400; unreferenced deletion removed artifact and registry. `ISSUE-003` was traced to an incomplete test fixture and corrected. |
| `S27` | Successful dataset validation | `PASS` | Full-dataset validation on the eight-row synthetic fixture completed twice with text, image, and pixel-distribution metrics; the recheck retrieved the persisted report (`104` total words, `20` unique words, `64x64` image dimensions). Synthetic technical scope only; the report `artifacts` map was empty, so separate artifact-file generation was not validated. |
| `S28` | Successful checkpoint evaluation | `PASS` | 2026-09-25: the pinned BEiT revision was restored in an isolated resource root, one CUDA epoch produced a checkpoint, and the recheck evaluated the retained checkpoint against the S27 fixture and retrieved the persisted report (`loss=2.0254554748535156`, `accuracy=0.8571428656578064`) with reload persistence. Synthetic technical scope only. |

### Tier 3 — Inference

| Slice | Capability | Status | Evidence scope and remaining gap |
| --- | --- | --- | --- |
| `S30` | Inference catalogue and public/custom separation | `PASS` | The current catalogue has four public entries after CXRMate-ED retirement; unit/API coverage verifies public/custom identity and retired-ref rejection. Catalogue cardinality is not a product invariant; live custom listing remains unobserved. |
| `S31` | Lightweight public-model installation lifecycle | `PASS` | 2026-09-24 isolated exact-pinned CXRMate Multi install: remote install/staging, one-file corruption and repair, real inference readiness promotion, stop/restart reuse with no redownload, deletion, and persistent inference history. The later canonical mismatch was repaired under `ISSUE-006`. PASS is limited to the fresh isolated install and one public image. |
| `S32` | Inference input and browser-state validation | `PASS` | Six live API E2E tests cover model context, all profiles, current-image count, image type/emptiness, invalid profile, and aggregate `64 MiB + 1 byte` rejection (HTTP 413). Rendered UI confirmed model-dependent context controls, profile changes, and the disabled generate state; no generation job was started in this slice. |
| `S33` | Complete browser inference workflow | `PASS` | 2026-09-24 rendered route: the public fixture was uploaded and exact revision `330721b9aa5bba201a3eb88eba4dd9a6607f3e7a` executed in the isolated runtime; generate, cancel, retry, persisted history, edit/save/reload, provenance, copy, and exported Findings/Impression content passed. The export download event was not surfaced by the browser adapter. One model and one public image only; `validation_status=pending`; no clinical or quality claim. |
| `S34` | Custom XREPORT checkpoint inference | `PASS` | 2026-09-25: the S28 checkpoint was listed as a ready custom model, selected in the rendered catalogue, and used for one synthetic multipart inference; persisted history and expanded provenance identified `xreport:XREPORT_20260925T141533`, `keras_checkpoint`, `fixed_224`, and `xreport_beit` across reload. Technical synthetic scope only. |
| `S35` | CXRMate-ED qualification and retirement | `RESOLVED` / retired | The bounded rescue replayed pinned revision `68251c7605067ddbea330413aade032713fd2192` with the published `max_length=256`, `num_beams=4` contract against three approved fixtures and produced one identical full report; the separate reference process was blocked by the unavailable pinned runtime. CXRMate-ED was removed from the active catalogue and adapter set; history, provenance, and the downloaded snapshot remain preserved. |
| `S36A` | CheXOne technical and broader quality gate | `PASS` | Pinned revision `0c350e6852ea08f9d9baf3b7595c1a10d4849927`: three manifest cases produced non-empty Findings-only output with correct provenance, no timeout/OOM/runtime error, measured CUDA/resource use, a reload/reuse sentinel, and a conservative manual review. Research-only technical evidence, not diagnostic-accuracy evidence. |
| `S36B` | CXRMate-2 technical and broader quality gate | `PASS` | Pinned revision `aa8e2d16470e20671acf049687b4707c9bf2f2b5`: the same three-case run produced non-empty Findings and Impression output with correct provenance, very-high resource use, a reload/reuse sentinel, and a conservative manual review. Research-only technical evidence. |
| `S37` | MedGemma gated-provider behavior | `PASS` | The gated-provider contract passed without model access: the live catalogue reports `access_policy=gated`, `not_installed`, the terms message, and the access URL; `generate_reports` rejects a gated non-ready public model before job start (HTTP 409); the rendered UI shows `NOT INSTALLED` and a disabled `Access required` Generate button. `ISSUE-002` remains open for authorized access. |

### Tier 4 — Heavy Providers and Desktop

| Slice | Capability | Status | Evidence scope and remaining gap |
| --- | --- | --- | --- |
| `S40` | Tauri development shell | `PASS` | Native CPU matrix `10 PASS`, `0 FAIL`, `0 UNTESTED`: startup, six routes, refresh/back/forward, settings save/restart, keyboard/modal/focus, backend stop/error, backend retry/recovery, second-instance policy, close/cleanup, and controlled delayed readiness (`XREPORT is still initializing`, backend after 18 seconds, ready `Inference`). Source CPU-shell evidence only. |
| `S41` | CPU packaged desktop | `PASS` | The v3.1.0 CPU portable/MSI pair (source `301a5a510e494c501d6c25eb22d5baea824e936a`, `dirty_tree=false`) passed official artifact verification, packaged startup/readiness/serving, graceful close, and a full MSI install→launch→graceful-close→uninstall lifecycle on one elevated transaction. The v3.2.0 CPU packages later passed verification, smoke, native acceptance, and MSI lifecycle checks. |
| `S42` | CUDA packaged desktop | `PARTIAL` | Exact v3.2.0 CUDA packages (run `39056753392`) passed GPU-host acceptance on the NVIDIA RTX 3060: artifact verification, portable smoke, native routes, Training submit/cancel, active-work close/reopen, synthetic inference (`resolved_device=cuda:0`, `cuda_available=true`, `cuda_used=true`), graceful cleanup, and MSI uninstall. No GPU-less Windows host is available; CPU fallback remains untested by user direction and is not a primary release blocker. Keep `PARTIAL`; do not claim fallback or full slice closure. |
| `S43` | Windows launcher maintenance actions | `PASS` | Disposable data-root/cache/checkpoint/database/model/log cleanup and idempotent repeats, source FastAPI/Node tree termination, listener cleanup, packaged parent-child ownership with official `KillProcesses`, and confirmed `RemoveDesktopRelease` cleanup with preserved canonical `data/` fingerprints. Launcher maintenance only; packaged native WebView, no-GPU, and performance boundaries remain outside. |

### Tier 5 — Resilience and Baseline

| Slice | Capability | Status | Evidence scope and remaining gap |
| --- | --- | --- | --- |
| `S50` | Restart and corruption recovery | `PASS` | Settings seed persisted across an ordinary stop/relaunch and rendered; injected `datasets` schema drift failed startup closed while the frontend stayed available with unavailable/retry states and the database unchanged; byte-for-byte backup restore recovered the schema/seed; post-restore health, API readback, and integrity passed. |
| `S51` | Concurrent operations and resource contention | `PARTIAL` | Source-runtime matrices passed: an eight-row CPU matrix, a real scale-8/64-row CPU contention matrix, an explicitly emulated backend-process unavailable-GPU matrix, repeated clean-restart matrices (`15` jobs across concurrent validation/evaluation/training triples with matched CPU/CUDA one-epoch baselines), and v3.2.0 native Training submit/cancel plus active-work close/reopen on packages. Open: genuine GPU-less hardware, hosted-CI concurrency, representative data, and the original slow-initialization attribution. Keep `PARTIAL`; package acceptance is not a full resilience pass. |
| `S52` | Accessibility and responsive workflow audit | `PASS` | The rebuilt source client passed the focused browser suite across Inference, Reports, Dataset, Training, and Settings at `320x640`, `390x844`, `640x800`, `1024x720`, and `1440x900` with 25 dimension-verified screenshots: route semantics, accessible names/labels, ARIA references, keyboard modal/tour behavior and focus restoration, settings tab navigation, training disclosure keys, reduced motion, responsive reflow, and bounded catalogue scrolling. Spoken screen-reader output was not observed and is non-gating. Source/browser scoped because the protected dist could not be overwritten. |
| `S53` | Performance and long-operation baseline | `PASS` | Final source revision `a10de59`: the deterministic scale-8, 64-row fixture ran through the real import/process path, then the ten-scenario matrix twice (three matched one-epoch CUDA runs, three matched one-epoch CPU runs, one three-epoch CUDA run, three inference repeats) including a clean-restart repeat with the exact tokenizer forced offline; receipts preserve timing, memory, GPU, terminal states, and warnings; device provenance was persisted; cleanup was clean and SQLite integrity remained `ok`. Source-runtime technical baseline only; packaged/no-GPU/clinical evidence remains outside. |

## Decisions and Supersessions

- **Scope boundaries.** Tier 5 covers source-runtime technical validation;
  packaged desktop and native WebView behavior remain Tier 4 concerns. Spoken
  Narrator/Speech Recap output is a separate non-gating accessibility follow-up
  and does not lower `S52` or any other gate status. `S02` timing is
  controlled-clock E2E evidence, not a wall-clock outage; packaged CPU/CUDA
  startup stays outside it.
- **v3.2.0 package supersession (2026-10-17/18).** The retained package set
  from run `39028661915` was invalidated by native review findings (`ISSUE-007`
  malformed Training wizard layout, `ISSUE-008` misleading cancellation
  message). Both were fixed in source
  `a8d75a0c0d5f02a29c85e9ac58f4837fc3081449`; replacement run `39056753392`
  passed verification, smoke, native acceptance, and MSI lifecycles and was
  approved and published on 2026-10-18.
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
  `ISSUE-006` (canonical CXRMate Multi manifest restored through the supported
  repair workflow on 2026-10-01) are resolved. `ISSUE-007` and `ISSUE-008` are
  fixed and confirmed in the exact v3.2.0 packages.

## Release Validation

The current release record is
[`summary-20261017.md`](../QA/validation_campaign/release-3.2.0-20261017/summary-20261017.md);
the approved hash binding is
[`approved-release-manifest.json`](../QA/validation_campaign/release-validation/approved-release-manifest.json).

- v3.2.0 was published 2026-10-18 by release workflow run `39103683278` from
  source `a8d75a0c0d5f02a29c85e9ac58f4837fc3081449`. The tag resolves to that
  source, all eight published asset digests match the approved manifest, and
  the release workflow passed CPU/CUDA artifact revalidation and portable
  smoke.
- Package run `39056753392` produced the accepted artifacts. The desktop
  release-gate receipts remain tracked under `assets/QA/desktop/`
  (`runtime-*`, `verification-*`, and `smoke-*` for CPU and CUDA 3.2.0).
- Acceptance: CPU portable, CUDA portable (GPU host), CPU MSI, and CUDA MSI
  passed native routes, Training submit/cancel, active-work close/reopen, and
  pinned synthetic inference; CPU resolved `cpu` with CUDA unavailable and
  unused, while CUDA packages resolved `cuda:0`. The final authorized CPU MSI
  uninstall on 2026-10-18 returned exit code `0`; no product registration,
  install tree, shortcut, app process, or listener remained.
- Remaining: `S42` and `S51` stay `PARTIAL`, and CUDA fallback on GPU-less
  hardware is untested by user direction and is not a primary release blocker.

## Evidence Policy

- The durable per-slice record is a compact summary at
  `assets/QA/validation_campaign/<slice>/summary-YYYYMMDD.md`. A summary
  records the exact Git revision and dirty-tree state; operating system,
  hardware, runtime variant, and database backend; commands and scenario IDs;
  the result (`PASS`, `PARTIAL`, `FAIL`, `BLOCKED`, `UNTESTED`, `UNKNOWN`);
  issue references, fixes, and adjacent regressions; hashes or durable paths
  for key receipts; and explicit omissions with the next action.
- Tracked evidence is limited to compact durable records: per-slice summaries,
  the approved release manifest, and the desktop release-gate receipts under
  `assets/QA/desktop/`. Screenshots, raw logs, fixtures, and bulky receipts are
  not tracked; keep them under `runtimes/cache` or local QA directories.
- When a tracked campaign directory is consolidated, its captures remain
  recoverable from Git history; for example the `release-3.2.0-20261017`
  acceptance captures and fixtures were removed from the working tree on
  2026-10-18 after their results were recorded.
- A durable `PASS` must not rest on transient-only output. Screenshots are
  required for visible browser or desktop acceptance; backend slices require
  request/response and log evidence.

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
