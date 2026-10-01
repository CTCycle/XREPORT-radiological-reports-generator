# Windows test runner and cache validation

Date: 2026-09-24
Branch: `develop`
Application/test base revision: `769fcd9b20e18c1bc8338ce73e55517d564b13a6`
Working-tree change during final run: test-only retrying assertions in `app/tests/e2e/test_preparation_filesystem_ui.py`.

## Scope and result

Executed the official `app/tests/run_tests.bat` from the repository root. It started the local backend and frontend on ports 5003 and 8003, ran the complete Python test target, the Angular unit suite, and the Angular E2E suite, then cleaned up services it had started.

The first run exposed one test-only timing failure in S15: after `page.goto()` and `page.reload()`, the test immediately sampled the image-folder button while `DatasetPage.refresh()` was still fetching its asynchronous status. Its initial `canBrowse()` value defaults to enabled until that response arrives. The API and user-facing behavior were otherwise correct. The original failure is preserved in [the first run log](full-run.log).

The test now uses Playwright's retrying `expect(...).to_be_disabled()` and `expect(...).to_be_enabled()` assertions. The complete rerun passed:

- Python: 191 passed, 3 skipped in 111.83 seconds. The skipped tests are the PostgreSQL persistence checks because no PostgreSQL service was configured.
- Angular unit: 40 passed across 12 files.
- Angular E2E: 14 passed.
- Runner summary: live server, Python, frontend bootstrap, frontend unit, and frontend E2E phases all `PASS`.

See [the final runner log](full-run-after-s15-test-fix.log). The S15 rendered flow covered filesystem access disabled/enabled, invalid path feedback and recovery, an empty folder, and selection of one image. The disabled and recovered states were visually reviewed in the saved [browser screenshots](s15-browser/).

## Cache and cleanup observations

The runner reported pytest's cache at `runtimes/cache/pytest`; its per-run basetemp was isolated under `runtimes/cache/pytest-tmp`. Both runner logs contain no pytest cache or permission warnings. `netstat.exe` showed no listeners remaining on ports 5003 or 8003 after cleanup.

Git directory scans still emit access-denied warnings for some historical cache paths elsewhere in the checkout. Those paths were not cleaned or modified because their ownership/access is unresolved. This leaves `test.infrastructure.windows_cache` and `ISSUE-004` `PARTIAL` / `OPEN`: the supported runner and its configured cache roots are validated, while safe disposition of the old protected cache paths remains outstanding.

The pushed validation commit `15b45a5d276a9ebf0453b0989fb92e39e404025f` also passed [hosted CI run 36014118210](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/runs/36014118210), including PostgreSQL persistence and every configured backend/client gate. GitHub emitted platform notices that Node 20 is deprecated and `ubuntu-latest` will migrate to Ubuntu 26 beginning 2026-10-19; no gate failed.

No application source change was needed. The three PostgreSQL checks were skipped only in the local Windows runner; hosted CI exercised that contract.
