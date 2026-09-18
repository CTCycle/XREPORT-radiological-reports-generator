# Testing And Quality Rules

Last updated: 2026-09-18

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

## Documentation And Change Discipline

- When behavior, contracts, or runtime modes change, update the corresponding docs in `assets/docs` in the same change.
- Keep changes scoped. Do not refactor unrelated modules unless explicitly required.
