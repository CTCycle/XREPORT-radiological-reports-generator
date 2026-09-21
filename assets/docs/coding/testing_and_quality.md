# Testing And Quality Rules

Last updated: 2026-09-21

## Tooling And Quality Gates

- Linting and formatting use Ruff or the established project-equivalent toolchain.
- Typing should remain Pylance-compatible.
- Testing uses pytest with coverage focused on `tests/unit` plus impacted `tests/e2e`.
  The Angular client also exposes `npm run test:e2e`, which runs the rendered
  UI checks in `app/tests/e2e/test_angular_ui.py` against the local services;
  `app/tests/run_tests.bat` starts those services when needed.
- Pytest configuration is centralized in `app/server/pyproject.toml`; CI and
  repository test helpers pass that file explicitly instead of relying on a
  root-level pytest configuration file.
- Validate changed API contracts and job-lifecycle behavior with targeted tests.
- Run `cargo check` from `app/desktop/src-tauri` for the Rust desktop shell;
  `build.rs` must always invoke `tauri_build` so the checked-in capability
  files are available to `tauri::generate_context!()` during debug builds.
- Keep `app/shared/openapi.json` synchronized with `app.server.app:app` and
  regenerate Angular types with `npm run generate:api` after contract changes.
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

Every executed slice must record the exact revision, dirty-tree state,
environment, scenarios, result, regressions, issue references, and durable
evidence under `assets/QA/validation_campaign/`. A passing unit or build gate
does not promote a feature workflow to end-to-end `VALIDATED`; the workflow must
also be exercised and observed at the appropriate boundary.

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
