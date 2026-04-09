# Testing Guide (XREPORT)

Last updated: 2026-04-08

This document describes the current test layout and execution flow.

## 1. Test Strategy

XREPORT uses three test scopes:
- `tests/unit`: fast unit tests for backend logic and contracts.
- `tests/e2e`: API and UI end-to-end tests (pytest + Playwright).
- `tests/backend/verification`: focused backend verification scripts.

Primary tools:
- `pytest`
- `pytest-playwright`
- Playwright request/page contexts

## 2. Current Test Layout

```text
tests/
- conftest.py
- run_tests.bat
- spaserver.py
- unit/
  - test_database_mode_env_override.py
  - test_e2e_training_fixture_paths.py
  - test_evaluation.py
  - test_job_cancellation_semantics.py
  - test_orm_data_access_refactor.py
  - test_pandas_string_normalization.py
  - test_training_memory_guards.py
  - test_training_stop_mechanism.py
- e2e/
  - perf_config.json
  - test_app_flow.py
  - test_database_api.py
  - test_inference_api.py
  - test_training_api.py
  - test_training_perf.py
  - test_upload_api.py
  - test_validation_api.py
- backend/verification/
  - verify_loader.py
  - verify_serializer.py
```

## 3. Running Tests

### 3.1 Recommended on Windows
```cmd
tests\run_tests.bat
```

Behavior:
1. Requires existing `runtimes/.venv` (created by `XREPORT/start_on_windows.bat`).
2. Reads host/port and optional settings from `XREPORT/settings/.env`.
3. Starts backend/frontend only if not already running.
4. Runs `pytest tests -v --tb=short`.
5. Cleans up only the services started by the script.

### 3.2 Manual execution
From repository root:
```powershell
runtimes\.venv\Scripts\python.exe -m pytest tests -v --tb=short
```

If backend/frontend are needed for specific E2E paths, start them first.

## 4. URL and Environment Resolution

`tests/conftest.py` resolves URLs in this order:
1. `APP_TEST_FRONTEND_URL` / `APP_TEST_BACKEND_URL`
2. `UI_HOST` + `UI_PORT`, `FASTAPI_HOST` + `FASTAPI_PORT`
3. Fallbacks (`127.0.0.1:7861` UI, `127.0.0.1:8000` API)

`0.0.0.0` is normalized to `127.0.0.1` for test requests.

## 5. Performance Tests

`tests/e2e/test_training_perf.py` is opt-in and skipped by default.

Enable it with:
- `RUN_PERF_TESTS=1`

Optional config path:
- `PERF_TEST_CONFIG_PATH=<path-to-json>`

Default config lookup is `tests/settings/perf_config.json` when present.

## 6. Writing New Tests

- Add unit tests under `tests/unit`.
- Add API/UI integration coverage under `tests/e2e`.
- Keep tests deterministic and isolated.
- For job-based APIs:
  - assert start response shape
  - poll `.../jobs/{job_id}`
  - assert terminal state and payload

Avoid brittle timing assumptions.

## 7. Troubleshooting

- Connection errors:
  - confirm backend/frontend URLs and ports.
  - prefer `127.0.0.1` on Windows.
- Missing Playwright/pytest dependencies:
  - ensure `runtimes/.venv` includes optional test dependencies.
- Stateful failures:
  - ensure no conflicting long-running job is active before test run.
