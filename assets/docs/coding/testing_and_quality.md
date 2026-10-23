# Testing And Quality Rules

Last updated: 2026-10-22

## Tooling And Quality Gates

- Linting and formatting use Ruff or the established project-equivalent toolchain.
- Typing should remain Pylance-compatible.
- The compact suite has 40 test source files: 17 Python unit suites, 7 Python
  E2E suites, 2 integration suites, 13 Angular specs, and 1 built-server test.
  It currently collects 411 cases (330 Python, 72 Angular, 9 Node). Keep
  regression coverage focused on application behavior and contracts.
- `app/tests/run_tests.bat` is the full Windows runner. It starts the local
  services as needed, runs the Python suites once, then runs Angular unit tests.
  Browser E2E cases are already included in pytest and are not invoked twice.
- From `app/client`, `npm run test:e2e` routes the focused rendered UI suite
  through `app/tests/run_tests.bat`, which starts the needed services and skips
  the unrelated Angular unit-test phase.
- Pytest configuration is centralized in `app/server/pyproject.toml`; CI and
  repository test helpers pass that file explicitly instead of relying on a
  root-level pytest configuration file.
- Validate changed API contracts and job-lifecycle behavior with targeted tests.
- Run `cargo check` from `app/desktop/src-tauri` for the Rust desktop shell;
  `build.rs` must always invoke `tauri_build` so the checked-in capability
  files are available to `tauri::generate_context!()` during debug builds.
- Keep `app/shared/openapi.xreport.json` and `openapi.studio.json` synchronized with
  `create_app("xreport")` / `create_app("studio")` (`app/scripts/export_openapi.py`) and
  regenerate Angular types with `npm run generate:api` after contract changes.
- Studio end-to-end modules (`test_training_api`, `test_upload_api`, `test_dataset_workflow`,
  `test_preparation_filesystem`, `test_studio_ui`) run in their own pytest process with
  `XREPORT_APP=studio` against the Studio backend (`APP_TEST_STUDIO_BACKEND_URL`,
  `APP_TEST_STUDIO_FRONTEND_URL`); `run_tests.bat` and CI start both backends.
- Verify the generic `/api/jobs` resource and explicit upload/checkpoint
  identity behavior when changing long-running workflows.

## Windows Script Rules

- Maintain compatibility with existing CMD launcher and build scripts where those scripts are the operational entrypoints.
- Use PowerShell for advanced scripting and automation when needed.
- Keep environment variable names and path semantics consistent with current scripts.
- All disposable test and tooling state belongs under `runtimes/cache`; the
  Windows test runner passes `runtimes/cache/pytest-tmp` as pytest's basetemp
  and `runtimes/cache/pytest` as its cache directory.
- Synchronous tests that await application coroutines must use the shared thread helper in `tests.conftest` when Playwright's synchronous E2E plugin is enabled.

## Validation Campaign Gates

The long-term validation campaign is tracked in
[`validation_campaign_ledger.md`](../validation_campaign_ledger.md). Its first
hard gate is Tier 0 (Gate A):

`S00 → S01 → S03 → S02`

S00 restores the current CI baseline, S01 proves source backend and SQLite
readiness, S03 reruns the cheap deterministic quality and build gates, and S02
proves the Angular startup/recovery shell. Do not advance to feature workflows
while S00 remains red on the current revision, even when a different operating
system passes locally.

When validation changes a status or establishes an important limit, update the
existing [campaign ledger](../validation_campaign_ledger.md) or
[status ledger](../project_status_ledger.md) with the source revision, scope,
result, and remaining boundary. Link the relevant test, CI run, or required
release manifest; do not create a standalone summary for every slice. Keep
screenshots, logs, and working receipts under `assets/QA/`. Track only files
required by project tooling; the approved release manifest is required by the
release verifier. A passing unit or build gate does not promote a feature
workflow to end-to-end `VALIDATED`; observe it at the appropriate boundary.

Use repository-local disposable state for the campaign:

```powershell
app\server\.venv\Scripts\python.exe -m pytest -c app/server/pyproject.toml app/tests/unit -q --basetemp runtimes/cache/pytest-tmp/tier0-unit -o "cache_dir=runtimes/cache/pytest"
$env:APP_TEST_FRONTEND_URL = "http://127.0.0.1:8003"
$env:APP_TEST_BACKEND_URL = "http://127.0.0.1:5003"
```

If a failure is reproducible, use the sequence
**inspect → execute → observe → diagnose → surgically fix → retest → record**.
Capture the original failure before changing code, run the narrow regression
set after the fix, and keep unrelated cleanup out of the remediation.

## Documentation And Change Discipline

- When behavior, contracts, or runtime modes change, update the corresponding docs in `assets/docs` in the same change.
- Keep changes scoped. Do not refactor unrelated modules unless explicitly required.
