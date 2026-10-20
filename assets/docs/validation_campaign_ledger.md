# XREPORT Validation Campaign Ledger

Last updated: 2026-10-20

This ledger summarizes current validation gates, release evidence, and known
scope limits. The [project status ledger](project_status_ledger.md) owns
component status and active issues; architecture and workflow documents own
their respective contracts. This is not a per-run archive.

Validation here means technical software evidence for a stated scope. It does
not establish clinical validity, diagnostic performance, regulatory approval,
or suitability for patient care.

## Gate map

| Tier | Scope | Current result | Boundary |
| ---: | --- | --- | --- |
| 0 | CI baseline, source readiness, deterministic quality gates, rendered [startup](runtime/startup.md) | `PARTIAL` (`S00` CI pass, Sonar unresolved; `S01`, `S03`, `S02` pass) | Current Windows S01/S02/S03 checks passed. Tier 0 remains partial because the latest available SonarCloud scan failed on the parent source and no Sonar check is attached to the current non-source candidate. |
| 1 | UI state, settings, jobs, [persistence](architecture/persistence.md), filesystem controls | `PASS` (`S10`–`S15`) | Scope is the exercised local SQLite and PostgreSQL 16 contracts. |
| 2 | [Dataset, training, and evaluation workflows](operations/workflows.md) | `PASS` (`S20`–`S28`) | Technical workflows use synthetic or small datasets; this is not representative-data or clinical-quality evidence. |
| 3 | Catalogue, installation, browser inference, custom and public [models](runtime/local_inference_models.md) | Complete (`S30`–`S37`; `S35` retired) | `CXRMate-ED` was retired after collapsing distinct fixtures; MedGemma's gated behavior passed, but model access remains blocked. Model quality remains unvalidated. |
| 4 | Heavy providers and Windows desktop [deployment](runtime/deployment.md) | `PARTIAL` (`S40`–`S43`) | `S43` remains partial for broader packaged/live-service ownership; GPU-less CUDA fallback and cross-variant performance remain untested. The v3.2.0 release package scope is recorded below. |
| 5 | Recovery, concurrency, [accessibility](ui/experience.md), performance | `PARTIAL` (`S50`–`S53`) | `S51` remains partial for heavy-worker contention and hardware coverage. Browser accessibility checks passed; spoken Narrator/Speech Recap output and packaged performance were not observed. |

Run the Tier 0 sequence in order: `S00 → S01 → S03 → S02`. A failed hard gate
blocks claims about dependent workflows until the current source is repaired
and revalidated. Select the smallest relevant regression set for other changes;
see [testing and quality rules](coding/testing_and_quality.md).

## Current candidate: launcher and Tier 0

- The application-source candidate validated is
  `b98a9b60492b63fb41a904e3cff4815325b8d075` on `develop`; the current ledger
  updates do not change application source. Its
  [Backend And Client Validation run](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/39424015542/job/118631860560)
  passed backend unit tests, Ruff, Pyright, PostgreSQL persistence, retained
  API E2E, client build, built-server, lint, and unit checks. This exact-commit
  CI result complements, but does not replace, the Windows startup checks below.
- `S00` remains partial. The current candidate has only the successful GitHub
  Actions check; no SonarCloud check is attached to it. The last available
  [SonarCloud result](https://github.com/CTCycle/XREPORT-radiological-reports-generator/runs/118229363263)
  is a failure on parent source `d5c69560af57aeacb6cf5a342cc5c8834cf3f849`
  with Security Rating `E` and Reliability `C` on New Code. The current
  candidate changes `.gitignore` and documentation only, so that source scan
  remains relevant but has not been cleared by a new Sonar run. Its check has
  no inline annotations; the issue view includes findings across unchanged project
  files. The project-wide Sonar gate remains unresolved.
- `S01` passed on the current Windows candidate using a disposable
  `XREPORT_RESOURCES_DIR`: fresh SQLite startup returned health `200`, reached
  Alembic head `e91a4f6c2d73`, created the required resource directories, and
  passed SQLite integrity. A restart reused the same 12-table schema. A
  non-empty database without `alembic_version` exited nonzero and was not
  stamped.
- `S03` passed locally on the current candidate: Ruff and Pyright passed;
  all 191 backend unit tests passed; after `npm ci`, the Angular production
  build and lint passed, as did all 53 client unit tests. The backend test
  commands were rerun with an absolute cache path under `runtimes/cache`.
- `S02` rendered startup and recovery checks passed in Chromium. The existing
  repository `serve-built.cjs` process already owned port 8003 and was left
  untouched; an isolated backend on port 5003 served the rebuilt bundle. Both
  browser tests passed: feature requests stayed gated through failed health
  probes, the shell showed progress and unavailable/retry states, retry
  recovered to the workspace, and a later catalogue error remained in-app.
  Captures are stored under ignored `assets/QA/validation_campaign/tier-0/`.
  The official launcher start path was not rerun because the existing UI
  process occupied its configured port, so launcher startup and process
  ownership remain open.
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
  and those instructions were corrected. `ISSUE-004` remains open because the
  pre-existing `app/server/runtimes/cache/pytest-xreport-current-validation`
  and `app/server/runtimes/cache/s23-20260924/pytest-cache` trees deny
  enumeration. They were left untouched.

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
