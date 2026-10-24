# XREPORT Validation Campaign Ledger

Last updated: 2026-10-24

## Browser validation plan (2026-10-24, run `xr-20261024-01`)

- V00–V03 passed on Windows against source `04820a7fdda5d9a36072fc0f54163037d366f894` and isolated SQLite roots. Hosted CI run [40114328827](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/40114328827) passed; local XREPORT and Studio builds/lints and all 79 Angular tests passed. The XREPORT live Chromium UI suite passed 17/17. The Studio runner preview could not bind its test URL (`8013`), so the same 7 Studio UI cases were run directly against the official Studio launcher and passed 7/7.
- V01 verified simultaneous XREPORT/Studio launch, isolated profile routes and databases, independent theme preferences, and fail-closed duplicate-resource-root locks. V02 verified Studio seed persistence across reload, XREPORT database-candidate apply/revert, and restored the original SQLite path across an official launcher restart.
- V03 used synthetic files only. The clean CSV loaded 8/8 rows; malformed and missing-text CSVs were rejected; partial matching required explicit confirmation and imported 7/8 rows. The viewer navigated records and retained the full long source path in its title while truncating the displayed cell. The S23 browser cleanup case now creates its own deterministic PNG/CSV fixture under `tmp_path` instead of depending on an ignored `s27` directory; focused Studio Chromium run passed 1/1.
- V04 processed the clean synthetic 8-row source to 7 train/1 validation rows, generated pixel/text/image statistics, and reopened the saved report after a page reload. A missing synthetic image was rejected before job creation. The UI accepts validation splits through `1.0`, while the backend schema caps them at `0.5`; a split of `1.2` produced the backend's typed 422. This is an open UI finding, recorded as `ISSUE-003`; no production code was changed. Evidence is in `assets/QA/validation_campaign/xr-20261024-01/v04-processing.log` and the run's screenshots.
- V05 is `BLOCKED`: the Standard CPU profile and local tokenizer are usable, but the pinned BEiT encoder is absent at `data/models/XRAYEncoder`. No weights were fetched. V06 and V07 are `BLOCKED` on the UI-trained V05 checkpoint.
- V08 is `PARTIAL` on XREPORT source `9b298c54`: the UI rejected a lone `.keras` file and a four-core manifestless ZIP with typed HTTP 400 responses before job admission, and the catalogue remained at zero custom checkpoints. The import dialog's copy mismatch is confirmed as `ISSUE-004`; checksum/version/conflict paths, positive import, cancellation and restart recovery require a V05 checkpoint.
- Gate map for this campaign (the exact runtime SHA is shown; `PARTIAL` retains any recorded finding or limit):

  | Gate | Status | SHA / branch | Evidence and limit |
  | --- | --- | --- | --- |
  | V00 | `PASS` | `04820a7f` / `develop` | Hosted CI 40114328827; builds, lints, 79 Angular tests; XREPORT Chromium 17/17; Studio Chromium 7/7 against official launcher after runner preview 8013 failed. |
  | V01 | `PASS` | `04820a7f` / `develop` | Isolated dual launch, profile routes, theme independence, duplicate-root locks; no package scope. |
  | V02 | `PASS` | `04820a7f` / `develop` | Seed and database-candidate persistence across reload/restart; isolated SQLite only. |
  | V03 | `PASS` | `04820a7f` app UI; `919e3b61` S23 harness | Synthetic upload/matching/viewer paths; self-contained S23 focused browser test 1/1. |
  | V04 | `PARTIAL` | `04820a7f` / `develop` (product code; live UI) | Processing and all validation metrics pass; UI/backend validation-split bounds mismatch is open as `ISSUE-003`. |
  | V05 | `BLOCKED` | `919e3b61` / `develop` | Pinned encoder absent; no model weights fetched. |
  | V06 | `BLOCKED` | `9b298c54` / `develop` | Requires a V05 UI-trained checkpoint for format-2 export/import. |
  | V07 | `BLOCKED` | `9b298c54` / `develop` | Requires V06 imported custom model for real XREPORT inference. |
  | V08 | `PARTIAL` | `9b298c54` / `develop` | UI lone-`.keras` and manifestless four-core ZIP rejected before job; no custom entry. Further archive cases depend on a V05 checkpoint. Dialog mismatch is `ISSUE-004`. |
  | V09–V10 | `UNTESTED` | — | Independent lifecycle, accessibility, responsive and performance checks remain. |
  | V11 | `UNTESTED` | — | Exact final-SHA rerun and ledger close remain. |
- Detailed logs and screenshots remain under ignored `assets/QA/validation_campaign/xr-20261024-01/`. V00–V08 changed no production code. Later rows will be updated here as their gates finish.

This ledger summarizes current validation gates, release evidence, and known
scope limits. The [project status ledger](project_status_ledger.md) owns
component status and active issues; architecture and workflow documents own
their respective contracts. This is not a per-run archive.

Validation here means technical software evidence for a stated scope. It does
not establish clinical validity, diagnostic performance, regulatory approval,
or suitability for patient care.

## Shim and legacy cleanup (2026-10-23, `develop` at `25561838` plus working tree)

- Scope: remove the browser theme-storage migration, manifestless/v1 checkpoint
  imports, Studio data-copy utility, its launcher action, and the old single-case
  inference-validation CLI path. Alembic history was not changed.
- PASSED (Windows): 300 XREPORT Python tests, 3 skipped; 26 Studio E2E tests;
  72 Angular tests; 9 built-server Node tests; Pyright; focused Ruff excluding
  existing complexity checks and ESLint on changed files.
  The app-lifespan lock test was deselected while the live backend owned its
  resource root, then passed alone after the services stopped.
- The inference draft E2E now waits for the asynchronously selected model to
  enable multi-image input before supplying two files. The earlier live
  checkpoint import used format version 1, so current version 2 live and
  packaged import remain unverified.
- Whole-project lint still reports complexity in the retained inference
  validation helpers and three array-type issues in the untouched inference
  draft spec.

## Persistent inference draft editing (2026-10-23, develop)

- Scope: XREPORT inference and report review. Successful job results expose the
  persisted request ID; the inference draft loads canonical history before
  enabling edits, saves only changed report rows, retains the original generated
  output, and links to Reports for recovery. Study-level output maps to its one
  persisted row. No Studio or persistence-schema changes.
- PASSED (Windows): Angular client suite, 13 files and 73 tests, using a
  temporary one-worker Vitest configuration after the default worker run hit a
  Windows temp-directory rename (`EPERM`). `npm run build` passed. Focused
  inference backend module: 21 passed, 1 deselected. History CRUD API E2E: 1
  passed. Browser edit/save/navigation/reload E2E: 1 passed, using a seeded
  SQLite history record and an intercepted deterministic generation job.
- The inference backend module's existing
  `test_inference_api_returns_conflict_for_a_degraded_public_model` was
  deselected after its Windows `TestClient` run hung; this is an environment
  limitation, not a confirmed product failure. Real-provider generation was
  not exercised in this campaign. The browser run validates persistence and
  recovery through the real history API, not model output quality.

The v3.2.1 candidate at
`1a15d24cb07fecbf66601940752a164c47972fc7` passed hosted CI run
[39667709442](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/39667709442).
That run passed backend unit tests, Ruff, Pyright, the PostgreSQL persistence
contract, retained API E2E, client build/server, lint, and client unit tests.
The candidate repairs upgrades from published database revisions
`f48a7c2e91b6` and `e91a4f6c2d73` and retains the v3.2.0 browser theme
preference. Its 16 focused database-initialization tests, four version/OpenAPI
tests, 65 client unit tests, and both lint gates pass locally. Current-source
navigation and seed persistence across a launcher restart passed with isolated
resources, with backend version 3.2.1. Official packages remain unverified.
Live authenticated PostgreSQL switching remains unverified.
Removed the ignored `.scratch-db-config-tests`, `.scratch-db-switch`, and
`.scratch-openapi` QA directories after consolidating their durable outcomes in
the existing ledgers. Referenced validation captures and the logs supporting
unresolved checks were retained.

## Application split (2026-10-23, branch `feature/split-studio`)

- Scope: XREPORT / XREPORT Studio separation, `.xreportckpt` exchange, Studio launcher, migration script.
- PASSED (Windows, source mode): backend unit + integration suite (including new profile, exchange, lock, migration, packaging-boundary and launcher tests); Ruff and Pyright on `app/server`; client build for both Angular projects, lint, and 66 unit tests; XREPORT inference/settings API e2e and `test_angular_ui` (23 tests) against a live XREPORT; Studio upload/training/dataset/preparation API and UI e2e (19 tests) plus `test_studio_ui` (7 tests) against a live Studio; both servers running together with isolated data folders; app-scoped kill; stale-build rebuild of only the changed project.
- PASSED (live, negative paths): import of a lone `.keras`, a traversal archive, a legacy archive without the encoder installed, and a crafted archive with a broken model were all rejected with explicit errors, with no checkpoint registered and the staging folder removed.
- PASSED (live, real model): with the encoder and tokenizer installed in a disposable data root, a tiny checkpoint registered in Studio was exported with the bundled encoder, downloaded, and imported into a separate XREPORT data root without an encoder; the model became `ready`, a report was generated from a synthetic image, history survived a restart, a repeat import returned `already_imported`, a mid-flight cancel and a mid-import process kill left no checkpoint, and the next start removed the staging folder. The checkpoint came from Studio's model builder, not from a UI training run.
- NOT RUN: Studio journey through the UI with real training, cancel/resume and evaluation; import into a packaged CPU or CUDA build; package contents audit; migration of real data; `run_tests.bat` end to end with the two-server setup.
- Facts versus claims: the XREPORT bundle isolation and the import-closure result are observed; the claim that packaged builds exclude Studio code follows from the spec and is untested until a package is built.
- PR #9 review fixes: source database configuration is checked before migration (PostgreSQL refusal and configured SQLite path); training/export start is mutually exclusive and deletion holds the job-registration lock. The 41 focused exchange, migration, and job-lifecycle tests, full backend unit suite (272 tests), Ruff, and full backend Pyright passed locally. The initial hosted run failed Pyright on the Keras output conversion; NumPy normalization repairs that error. SonarCloud reported 4.7% duplication, dominated by generated API types; `.sonarcloud.properties` excludes only those two generated files from duplication checks. Hosted CI passed at `0121c6b4` in [run 39931654917](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/39931654917): backend unit tests, Ruff, Pyright, PostgreSQL persistence, both profiles' retained API E2E, both client builds, built-server tests, client lint and unit tests. SonarCloud and GitGuardian also passed.

## Release 3.3.0 smoke (2026-10-22)

- Scope: source launcher on develop plus the portable-Node recovery fix from draft PR 8 (cherry-picked; hosted CI on the PR passed). Backend unit suite (232), integration/backend suites (4 passed, 3 skipped), Ruff, client lint, 65 client unit tests, production client build, and PowerShell parse check passed locally.
- Official `Launch` with isolated resources served `/api/health` (version 3.3.0 after bump, SQLite, Alembic head `42c8e1f7b0d3`), `/`, `/dataset`, `/training`, `/settings`, and `/api/inference/models` with HTTP 200 and the port closed afterward.
- Full standard runner on 3.3.0 (2026-10-22): 278 passed, 3 skipped, 2 failed in the Python phase; 65 client unit tests and 18 Angular browser E2E passed. Failure 1 (`S23`, missing `s27` fixtures) is `BLOCKED` by environment. Failure 2 (request-drain gate test) was a test defect: an earlier Playwright test left a running loop, so `asyncio.run` raised; the test now runs on a clean thread and passes in isolation, and the full runner rerun afterward (2026-10-22) passed it: 279 passed, 3 skipped, 1 failed (`S23`, same missing-fixture `FileNotFoundError`, `BLOCKED`); 65 client unit tests and 18 Angular browser E2E passed (`PASSED`).
- Real `Launch` (isolated resources): health 3.3.0 and all routes returned 200; `/training` rendered in the in-app browser with no console errors. `KillProcesses` reported a failure stopping PID 21860 but ports 5003/8003 were clear afterward Root cause reproduced: `Stop-XReportProcesses` checked for survivors immediately after `taskkill`, before the process finished exiting. Fixed in `start_on_windows.ps1` with a bounded wait (up to 5 s). Retest: `Launch` then `KillProcesses` exited 0, a second call reported nothing to stop, and ports 5003/8003 had no listeners (`PASSED`). Rendered `/dataset` and `/inference` loaded in the in-app browser after the startup splash with no console errors.
- Superseded shim-removal suite logs (full suite PASS) were consolidated here and deleted.
- Live E2E on `develop` at `a442133` plus the working-tree fixes (2026-10-22, isolated resources, source `Launch`, 3.3.0): restoring the git-history `s27` fixture pack (ignored path, untracked) cleared the S23 block, and the S21/S23 dataset UI/API and S15 filesystem E2E (6 tests) passed against the live app (`PASSED`). A scripted API journey then ran upload, load (8/8 matched), processing, a real 2-epoch training job (checkpoint `ready`, val_loss 3.95), dataset validation (image/text statistics), checkpoint evaluation (accuracy 0.71, BLEU 0.158), and custom-checkpoint inference on `cuda:0` with a saved history entry (all `PASSED`; synthetic 64x64 data, not quality evidence). The in-app browser rendered `/reports` and `/training` with the new session, checkpoint, and datasets and no console errors. `KillProcesses` then relaunch on the same resources kept the checkpoint, history, and processed datasets, with ports clear after each stop (`PASSED`).
- Environment finding: training loads the pinned BEiT encoder (`microsoft/beit-base-patch16-224`, `f02e8f77`) with `local_files_only`; on a machine without `models/XRAYEncoder` the job fails with a generic "couldn't connect to huggingface.co" message. This was an environment prerequisite, not a code defect; the encoder was downloaded with user approval for this run and removed afterward. Not changed in code; a clearer offline-encoder message is a possible improvement.
- Not re-run: live PostgreSQL switching, packaged-artifact smoke before publication, GPU-less CUDA fallback, MedGemma (`BLOCKED`, gated access). Existing limits in the gate map still apply.

## Release 3.3.1 patch (2026-10-22)

- Scope: patch over 3.3.0 containing the `KillProcesses` exit-wait fix and the request-gate test fix; no application, API, or UI changes. Backend unit suite (232) and Ruff passed locally; the 3.3.0 live E2E and launcher lifecycle evidence above covers the same source apart from the version bump (`PASSED`).
- Packages: tag `v3.3.1` at `7402615` built CPU and CUDA MSI and portable artifacts in workflow run 39810518968 (in-workflow runtime/MSI audit and packaged portable smoke passed). The artifacts were downloaded and re-verified locally with `verify_desktop_artifacts.ps1` and the packaged portable startup smoke for both variants (backend ready, frontend index served, clean process/listener shutdown; `PASSED`). Runtime audits were recovered from the portable payload manifests.
- Not run: MSI install, upgrade from 3.3.0, and uninstall (needs an elevated per-machine install; manual check required), packaged CPU/CUDA inference, GPU-less CUDA fallback, live PostgreSQL switching (`NOT RUN`). README screenshots were audited and left unchanged.

## Gate map

| Tier | Scope | Current result | Boundary |
| ---: | --- | --- | --- |
| 0 | CI baseline, source readiness, deterministic quality gates, rendered [startup](runtime/startup.md) | `PARTIAL` (`S00` candidate CI passed; current-source navigation and restart persistence passed; focused local migration/version/OpenAPI/client/lint checks passed; official packages remain open; rendered `S02` checks pass) | Candidate commit `1a15d24` passed hosted run 39667709442, including backend unit tests, Ruff, Pyright, PostgreSQL persistence, retained API E2E, client build/server/lint/unit tests. The official launcher used isolated resources on ports 5003/8003; Inference, Reports, Dataset, Training, and Settings loaded; seed `317` survived navigation and restart; backend version was 3.2.1, SQLite integrity passed, and Alembic reached `42c8e1f7b0d3`. CPU/CUDA packages and package smoke are still open. SonarCloud is excluded because the available plan does not support this repository's over-50k LOC analysis. |
| 1 | UI state, settings, jobs, [persistence](architecture/persistence.md), filesystem controls | `PARTIAL` (`S10`–`S15`; database runtime switch partial) | The v3.2.1 candidate restores the browser theme preference to database settings when the released database has no saved value; client unit tests cover import and stale-copy removal. SQLite migration tests cover both revisions from published v3.2.0 and preserve settings/data. Successful authenticated PostgreSQL connection and UI activation remain unverified; live PostgreSQL testing was deferred on 2026-10-21. Broader Tier 1 workflows retain their existing limits. |
| 2 | [Dataset, training, and evaluation workflows](operations/workflows.md) | `PARTIAL` (`S20`–`S28`) | S23 passes when the untracked `assets/QA/validation_campaign/s27/fixtures` pack is present (restorable from git history before `05b179f`); without it S23 is `BLOCKED`. Other workflows use synthetic or small datasets; this is not representative-data or clinical-quality evidence. |
| 3 | Catalogue, installation, browser inference, custom and public [models](runtime/local_inference_models.md) | Complete (`S30`–`S37`; `S35` retired) | `CXRMate-ED` was retired after collapsing distinct fixtures; MedGemma's gated behavior passed, but model access remains blocked. Model quality remains unvalidated. |
| 4 | Heavy providers and Windows desktop [deployment](runtime/deployment.md) | `PARTIAL` (`S40`–`S43`) | `S43` remains partial for broader packaged/live-service ownership; GPU-less CUDA fallback and cross-variant performance remain untested. The v3.2.0 release package scope is recorded below. |
| 5 | Recovery, concurrency, [accessibility](ui/experience.md), performance | `PARTIAL` (`S50`–`S53`) | The S51 deterministic job-admission and harness-contract unit checks passed on the current source; live heavy-worker/GPU contention and hardware metrics remain untested. Browser accessibility checks passed; spoken Narrator/Speech Recap output and packaged performance were not observed. |

Run the Tier 0 sequence in order: `S00 → S01 → S03 → S02`. A failed hard gate
blocks claims about dependent workflows until the current source is repaired
and revalidated. Select the smallest relevant regression set for other changes;
see [testing and quality rules](coding/testing_and_quality.md).

## Current candidate: launcher and Tier 0

- The follow-on compatibility cleanup removed the neutral
  `generation_profile_contract_version` field from validation contract hashes,
  so existing receipts with former hashes are stale and may require fresh
  validation. Launcher cache cleanup now targets only the canonical runtime
  cache; startup documentation reflects that scope. The client replaced
  `openapi-typescript` with pinned `@hey-api/openapi-ts` `0.99.0`, generates
  TypeScript types only, pins Angular tooling to `22.1.8`, and no longer bypasses
  npm peer checks. A clean npm install and dependency tree, generated API types,
  client production build/lint, 64 client unit tests, 18 dedicated Angular E2E
  tests, backend Ruff, and Pyright passed. The full standard runner collected
  252 tests: 245 passed, 3 skipped, and 4 failed. The failures were the missing
  S23 fixture, an S10 browser console 502 (its focused retry could not connect
  to the runner listener), a transient upload request reset (the focused retry
  passed), and the request-drain gate test failing before its body due to
  Windows asyncio loop initialization. The overall end-to-end result remains
  partial. A broader Ruff invocation that included tests reported nine
  complexity/statement-count findings in existing test files; the backend
  source Ruff gate passed.
- On 2026-10-21 the application source dropped its legacy compatibility layers:
  the Alembic stream was squashed to the single current-schema baseline
  `b1e7a2c4d908`, the database-configuration bootstrap stopped reading legacy
  `.env`/`DATABASE_*` values and now creates a default SQLite configuration,
  the former packaged settings-JSON cleanup was removed, and the client no
  longer migrates a browser-stored theme preference into the database.
  Post-change reruns on Windows: backend pytest excluding live-server E2E
  passed 205 cases with 3 skipped, the Angular production build, lint, and all
  64 client unit tests passed, the browser/UI E2E runner passed 18 cases
  against a rebuilt bundle, and the Python browser/API E2E suite passed 43
  cases; its single remaining failure is `S23`, blocked by the still-absent
  `assets/QA/validation_campaign/s27/fixtures` pack. Successful launcher
  current-source launcher startup on standard ports and rendered inference UI
  now pass. Current-source restart reuse and a fresh hosted CI run for the
  modified working tree remain open.
- Current working-tree verification on 2026-10-21, based on HEAD
  `ef1ae34ad8e9dc073415e5ec1a0eaec588f81999`: the official launcher started
  the rebuilt bundle and backend on ports `8003`/`5003` with an isolated
  `XREPORT_RESOURCES_DIR`; health and the rendered `/inference` page returned
  HTTP 200. Fresh SQLite integrity was `ok` at Alembic head
  `42c8e1f7b0d3`. The full backend unit suite passed 223 tests, Ruff and
  Pyright passed, client lint passed, all 64 Angular unit tests passed, and
  all 9 built-server tests passed. Local Node was `22.23.1`; the workflow pins
  `22.22.3`. One real local run completed for public CC0 chest-radiograph
  input using CXRMate Multi TF revision
  `330721b9aa5bba201a3eb88eba4dd9a6607f3e7a`; provenance reports
  `cuda:0`/CUDA used and the UI showed Findings and Impression for a public
  [CC0 radiograph from Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg). Catalogue
  status became `ready`, but `validation_status` is still `pending` and the
  receipt is missing, so this is technical execution evidence only. Screenshot:
  [live inference](../QA/codex-live-inference-20261021.png).
- The latest [GitHub Actions run for HEAD
  `ef1ae34`](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/39643961325)
  reported two PostgreSQL 16 persistence-contract failures caused by redundant
  reflected parentheses; Ruff, unit tests, Pyright, and the other persistence
  cases passed. The current working tree adds normalization and a regression
  case, and its unit suite passes. The live PostgreSQL contract was not rerun
  after the user deferred PostgreSQL testing; no hosted run can include the
  uncommitted working-tree changes.
- The application-source candidate validated before the legacy cleanup is
  `b98a9b60492b63fb41a904e3cff4815325b8d075` on `develop`; the CI and launcher
  evidence bullets below pinned to that revision. Its
  [Backend And Client Validation run](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/39424015542/job/118631860560)
  passed backend unit tests, Ruff, Pyright, PostgreSQL persistence, retained
  API E2E, client build, built-server, lint, and unit checks. This exact-commit
  CI result complements, but does not replace, the Windows startup checks below.
- `S00` passed: the exact application-source candidate has a successful
  [GitHub Actions run](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/39424015542)
  covering backend unit tests, Ruff, Pyright, PostgreSQL persistence, retained
  API E2E, client build, built-server, lint, and unit checks. SonarCloud is not
  a project gate: the available plan does not support analysis for this
  repository's over-50k LOC, so no Sonar result is required for S00.
- `S01` current-source launch/render passed on 2026-10-21 using an isolated
  `XREPORT_RESOURCES_DIR` on standard ports `5003`/`8003`: backend health and
  the rendered inference route returned HTTP 200, and fresh SQLite integrity
  passed at `42c8e1f7b0d3`. The current backend unit suite also passed the
  non-empty unversioned-database rejection case. Current-source process restart
  reuse remains unverified, so the broader S01 sequence is still partial.
- `S03` passed locally on the current candidate: Ruff and Pyright passed;
  all 191 backend unit tests passed; after `npm ci`, the Angular production
  build and lint passed, as did all 53 client unit tests. The backend test
  commands were rerun with an absolute cache path under `runtimes/cache`.
- `S02` rendered startup and recovery checks passed in Chromium. Both browser
  tests passed: feature requests stayed gated through failed health probes,
  the shell showed progress and unavailable/retry states, retry recovered to
  the workspace, and a later catalogue error remained in-app. Captures are
  stored under ignored `assets/QA/validation_campaign/tier-0/`.
- Settings follow-up validated on `ef1ae34ad8e9dc073415e5ec1a0eaec588f81999`:
  the rendered General, Data access, Advanced, and Database tabs were exercised
  in the in-app browser on isolated SQLite resources. Autosaved values survived
  reload and backend restart; bounds rejected out-of-range seed, polling, and
  timeout values; reset restored defaults; and separate SQLite files retained
  distinct settings. A failed PostgreSQL connection probe and an unconfirmed
  database draft left SQLite active and Apply disabled. `/api/settings` exposed
  only the user-facing inference timeout, not the private process policy.
  Four focused browser/API E2E cases and 14 focused client settings specs passed.
- The same revision's full backend unit suite passed (222 tests). An isolated
  `.env` fixture validated defaults, accepted boolean/device/list values,
  rejected invalid boolean/device values and wildcard origins with credentials,
  and exercised allowed and rejected CORS preflights. Empty CORS origins left
  CORS disabled. A SQLite database created at baseline `b1e7a2c4d908` upgraded
  through normal initialization to `42c8e1f7b0d3`; user settings and a dataset
  record were preserved, the former `inference_hf_local_only` and
  `inference_device` columns were removed, and repeat initialization succeeded.
  Backend Ruff and Pyright passed. The ignored local `settings/.env` remained
  read-only; only isolated fixture/source-level environment behavior is
  recorded here. Successful PostgreSQL activation remains unverified; the user
  explicitly deferred that test in the current follow-up, and the separate
  request-drain gate remains unverified.
- The official launcher then completed a full start on isolated ports
  `51873`/`41873` with a disposable `XREPORT_RESOURCES_DIR`. It detected the
  missing frontend build manifest, rebuilt the production bundle, and returned
  HTTP 200 from both the UI root and backend `/api/health`. The launched
  listeners and disposable data were removed afterward, and `settings/.env`
  was restored byte-for-byte. The configured UI port `8003` was still owned by
  the pre-existing repository `serve-built.cjs` PID `37384`; a default-port
  non-interactive launch failed closed and left that process running. Successful
  startup on the standard ports and process-ownership races remain open.
- Launcher behavior passed isolated disposable fixtures: single-package
  cleanup recalculated remaining checksums and release metadata; per-version
  cleanup removed only that version's payloads and stale sidecars; all-version
  cleanup removed matching release artifacts and generated desktop state while
  preserving unrelated files, source, QA, and user data. A throwing
  `Write-Progress` fallback completed cleanup. Valid, stale-source,
  stale-dependency, missing, invalid, and missing-output frontend manifests
  produced the expected states. The occupied-port non-interactive path failed
  closed; the interactive decline path was exercised with a disposable
  listener and simulated prompt input and preserved that listener.
- `git check-ignore` confirmed the launcher and test-runner cache paths are
  ignored, including `app/client/.angular/cache/`, while allowlisted source,
  QA evidence, and `runtimes/cache/.gitkeep` remain trackable. The configured
  Windows test runner uses `runtimes/cache` and a per-run pytest base-temp.
  Direct pytest instructions now resolve the absolute cache root; using a
  relative `cache_dir` had placed pytest state under `app/server/runtimes/cache`
  and those instructions were corrected. Launcher cache clear now operates
  only on canonical `runtimes/cache`, and uninstall no longer scans historical
  cache locations. Existing historical cache contents were not modified.

## Current release evidence

- Test-suite consolidation on the uncommitted working tree based on
  `1cc34750dce461408bee513f42015e0dfd7f294a` (2026-10-22) reduced executable
  test sources from 71 to 32 and collected cases from 357 to 355. The full
  Windows runner passed 281 Python tests with 3 skipped and 62 Angular tests
  in 162.86 seconds; the pre-change run took 310.91 seconds and repeated the
  18-case browser suite. The separately focused browser command passed all 18
  cases, and the CI-selected API E2E set passed 19. Integration tests passed
  4 cases and skipped 3; the PostgreSQL contract remains gated in local runs.
  The after timing followed standalone test and build checks that warmed the
  shared caches, so the full-run comparison is observed rather than normalized.
  Hosted CI was not run, and no production source was changed.
- v3.2.0 was [published](https://github.com/CTCycle/XREPORT-radiological-reports-generator/releases/tag/v3.2.0)
  on 2026-10-18 from source `a8d75a0c0d5f02a29c85e9ac58f4837fc3081449`.
  [Package run 39056753392](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/39056753392)
  supplied the accepted CPU and CUDA artifacts;
  [release workflow run 39103683278](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/39103683278)
  published them. The tag resolves to the source commit, all
  eight published SHA-256 digests match the [approved release manifest](../QA/validation_campaign/release-validation/approved-release-manifest.json),
  and release verification plus portable smoke passed for both variants.
- CPU and CUDA portable and MSI packages were exercised on an RTX 3060 Windows
  host. Native routes, Training submit/cancel, active-work close/reopen,
  synthetic inference, and install/uninstall lifecycles passed. CPU inference
  resolved to `cpu` with CUDA unused; CUDA inference resolved to `cuda:0`.
- The final CPU MSI uninstall returned exit code `0`; no product registration,
  install tree, shortcut, XREPORT process, or listener remained.
- This is a technical release scope. No GPU-less Windows host was available,
  so CUDA fallback remains untested. `S42` and `S51` remain `PARTIAL`; these
  gaps do not invalidate the narrower package results. Synthetic inference is
  not clinical or diagnostic-performance evidence.

## Durable decisions and limits

- `CXRMate-ED` remains retired: its bounded qualification run produced the
  same full report for three distinct fixtures. Do not restore it without a
  new independent qualification. See [local inference models](runtime/local_inference_models.md).
- The MedGemma gated-provider contract is validated, including rejection
  before job start and a disabled Generate action. Real installation and
  inference require authorized provider access; see `ISSUE-002` in the
  [status ledger](project_status_ledger.md).
- `S50` recovery, `S52` responsive/accessibility checks, and `S53` source
  performance baselines passed for their stated technical scopes. Packaged
  timing, genuine no-GPU fallback, broader representative-data behavior, and
  spoken accessibility remain outside those results.
- `S51` bounded concurrency recheck passed 15 tests across
  [`test_job_lifecycle.py`](../../app/tests/unit/test_job_lifecycle.py)
  and
  [`test_resilience_harness.py`](../../app/tests/unit/test_resilience_harness.py)
  on source candidate `b98a9b60492b63fb41a904e3cff4815325b8d075`. These checks
  cover atomic duplicate-start rejection, overlap and cleanup for distinct job
  types, and the resilience harness's scenario and worker-observation
  contracts with stubbed workers. The live S51 workload matrix was not run because its default
  `assets/QA/validation_campaign/s27/fixtures` directory is absent. Although
  this host exposes an RTX 3060 Laptop GPU, no real heavy worker was started;
  GPU contention, high-water resource samples, and real-worker cleanup remain
  unvalidated, so S51 stays `PARTIAL`.
- Public-model technical checks and output review do not show diagnostic
  accuracy. Keep model readiness and `validation_status` separate from any
  clinical or quality claim.

## Evidence and maintenance

- Keep current outcomes, source revision, scope, and limitations in this
  ledger or the component entry in the status ledger. Link existing tests,
  CI runs, architecture/workflow docs, or the release manifest where they
  support the claim; do not create a parallel report for each slice.
- Store screenshots, logs, and working QA reports under `assets/QA/`. Keep
  only artifacts required by project tooling in the tracked tree. The approved
  release manifest remains tracked because the release verifier consumes it
  and it binds one source commit to the published artifact hashes.
- Keep `PASS`, `PARTIAL`, `BLOCKED`, `FAIL`, and `UNTESTED` scoped to what was
  actually exercised. A unit test does not prove a browser workflow; a
  synthetic technical run does not prove model quality; package evidence does
  not prove behavior on hardware that was not tested.
- When a regression is found, record the affected scope, fix it narrowly, run
  the adjacent checks, and update the relevant status and limit. Do not retain
  superseded failures as active issues.
