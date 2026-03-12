# Testing Guide

This document describes the current test strategy for XREPORT and how to run or extend the suite.

## Overview

XREPORT uses a mixed test strategy:
- `tests/unit`: fast Python unit tests for core behavior (job manager semantics, config parsing, evaluation helpers, normalization utilities).
- `tests/e2e`: API + UI tests using Playwright + pytest against running backend/frontend.
- `tests/backend/verification`: focused verification scripts for serializer/loader checks.

Primary tools:
- `pytest`
- `pytest-playwright`
- Playwright browser automation + API request context

## Current Test Layout

```text
tests/
- run_tests.bat
- conftest.py
- unit/
  - test_database_mode_env_override.py
  - test_evaluation.py
  - test_job_cancellation_semantics.py
  - test_pandas_string_normalization.py
  - test_training_stop_mechanism.py
- e2e/
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

## Running Tests

### Recommended (Windows automation)
```cmd
tests\run_tests.bat
```

What this script does:
1. Validates prerequisites (`runtimes/.venv`, npm/runtime availability).
2. Starts backend/frontend if not already running.
3. Runs `pytest tests -v --tb=short`.
4. Cleans up services it started.

### Manual run
1. Start the app (`XREPORT\start_on_windows.bat`) or manually start backend + frontend.
2. Run:
```cmd
pytest tests
```

## URL and Environment Resolution

`tests/conftest.py` resolves URLs in this order:
1. explicit `APP_TEST_FRONTEND_URL` / `APP_TEST_BACKEND_URL`
2. `UI_HOST`/`UI_PORT` and `FASTAPI_HOST`/`FASTAPI_PORT`
3. fallbacks (`127.0.0.1:7861` UI, `127.0.0.1:8000` API)

It also normalizes `0.0.0.0` to `127.0.0.1`.

## E2E Scope

E2E coverage currently targets:
- UI navigation and page rendering (`/dataset`, `/training`, `/inference`)
- upload endpoint (`/upload/dataset`)
- preparation endpoints (`/preparation/*`)
- training endpoints (`/training/*`)
- inference endpoints (`/inference/*`)
- validation endpoints (`/validation/*`)

Long-running operations are validated through job start + polling endpoint behavior, not by full-length model training in default runs.

## Performance Tests

`tests/e2e/test_training_perf.py` is opt-in and skipped unless:
- `RUN_PERF_TESTS=1`

These tests are heavier, may require cached model assets, and can run with additional perf-related env configuration.

## Writing New Tests

- Unit tests: add under `tests/unit` and keep them deterministic and isolated.
- E2E API tests: use `api_context` fixture.
- E2E UI tests: use `page` fixture and stable selectors (`title`, ids, role/text where appropriate).
- For job-based APIs:
  - assert 202 start response shape
  - poll `/.../jobs/{job_id}` to terminal status
  - assert terminal payload (`result`/`error`) rather than timing assumptions

## Troubleshooting

- `ECONNREFUSED` or timeout:
  - verify backend/frontend URLs and ports.
  - prefer `127.0.0.1` over `localhost` on Windows.
- Playwright missing:
  - ensure optional deps are installed in `runtimes/.venv`.
- flaky stateful failures:
  - make sure no long-running training job is already active before running tests that require idle state.
