# XREPORT Validation Campaign Ledger

Last updated: 2026-10-21

This ledger summarizes current validation gates, release evidence, and known
scope limits. The [project status ledger](project_status_ledger.md) owns
component status and active issues; architecture and workflow documents own
their respective contracts. This is not a per-run archive.

Validation here means technical software evidence for a stated scope. It does
not establish clinical validity, diagnostic performance, regulatory approval,
or suitability for patient care.

The later working-tree configuration change adds environment-driven CORS,
moves inference process policy out of the application-settings table, and adds
Alembic migration `42c8e1f7b0d3`. It has no validation evidence yet; the prior
gate results below describe the source tested at the time.

## Gate map

| Tier | Scope | Current result | Boundary |
| ---: | --- | --- | --- |
| 0 | CI baseline, source readiness, deterministic quality gates, rendered [startup](runtime/startup.md) | `PARTIAL` (`S00`, `S01`, `S03`, and rendered `S02` checks pass; current-source launcher start remains open) | GitHub Actions and Windows S01/S02/S03 checks passed for the previously validated source. The official launcher completed a full start on isolated ports. A prior standard-port launch failed closed while preserving an existing listener; the current recheck found no listeners on ports 5003 or 8003, but the launcher has not been rerun on the current source. SonarCloud is excluded from the gates because the available plan does not support this repository's over-50k LOC analysis. |
| 1 | UI state, settings, jobs, [persistence](architecture/persistence.md), filesystem controls | `PARTIAL` (`S10`–`S15`; database runtime switch partial) | Non-database settings autosave, apply immediately where feasible, and restore after reload; database edits remain drafts until the current candidate passes a connection test and the user confirms. Existing SQLite and PostgreSQL 16 persistence contracts pass. The Database UI was exercised across two SQLite files, including per-database settings reload, restart persistence, unavailable-PostgreSQL handling, and rollback. A successful PostgreSQL connection and UI switch remain unverified because the local service requires credentials unavailable for this run. |
| 2 | [Dataset, training, and evaluation workflows](operations/workflows.md) | `PARTIAL` (`S20`–`S28`) | The current standard run includes an S23 failure because `assets/QA/validation_campaign/s27/fixtures/images` is absent. Other workflows use synthetic or small datasets; this is not representative-data or clinical-quality evidence. |
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
  startup on the standard UI port and a fresh CI run for this new source
  revision remain open.
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
- `S01` passed on the earlier Windows candidate using a disposable
  `XREPORT_RESOURCES_DIR`: fresh SQLite startup returned health `200`, reached
  the then-current Alembic head, created the required resource directories, and
  passed SQLite integrity. A restart reused the same 12-table schema. A
  non-empty database without `alembic_version` exited nonzero and was not
  stamped. Post-cleanup, fresh initialization reaches the squashed head
  `b1e7a2c4d908` under unit coverage and the local runtime recreated its
  database at that head; the full launcher S01 sequence has not rerun on the
  new source.
- `S03` passed locally on the current candidate: Ruff and Pyright passed;
  all 191 backend unit tests passed; after `npm ci`, the Angular production
  build and lint passed, as did all 53 client unit tests. The backend test
  commands were rerun with an absolute cache path under `runtimes/cache`.
- `S02` rendered startup and recovery checks passed in Chromium. Both browser
  tests passed: feature requests stayed gated through failed health probes,
  the shell showed progress and unavailable/retry states, retry recovered to
  the workspace, and a later catalogue error remained in-app. Captures are
  stored under ignored `assets/QA/validation_campaign/tier-0/`.
- The Database Settings UI passed an in-app browser workflow on isolated
  resources: connection tests and switches succeeded between two SQLite
  databases; each restored its own default seed (`777` and `42`); a backend
  restart retained the selected file and settings; and an unreachable
  PostgreSQL target produced an error while keeping SQLite active. The local
  PostgreSQL service required a password, so a successful PostgreSQL UI switch
  remains unverified. The theme selector also saved Dark in one SQLite file,
  restored System in a second, and restored Dark after switching back and after
  a backend restart.
- The settings autosave recheck passed 19 focused client unit tests and four
  Chromium E2E cases. It covered the General, Data access, Advanced, and
  Database tabs; debounced seed persistence across an immediate page reload;
  checkbox, polling, timeout, theme, and all database connection-field edits;
  reset; and an unconfirmed
  database draft whose failed PostgreSQL probe left SQLite active and kept
  Apply disabled. Client tests also covered serialized rapid writes, the
  current-draft validation gate, and recovery from a failed write.
  After a backend process restart on isolated resources, seed `321`, disabled
  filesystem access, polling interval `2.25`, timeout `777`, and Dark theme all
  restored from the active database. The application-settings repository
  reopen test passed 11 cases; database configuration validation and rollback
  passed 8 cases. A working PostgreSQL service was unavailable, so successful
  PostgreSQL activation remains open. The separate request-drain gate test did
  not reach its test body because Windows asyncio loop initialization stalled;
  that gate remains unverified in this run.
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
  [`test_job_start_concurrency.py`](../../app/tests/unit/test_job_start_concurrency.py)
  and
  [`test_validate_resilience_baseline.py`](../../app/tests/unit/test_validate_resilience_baseline.py)
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
