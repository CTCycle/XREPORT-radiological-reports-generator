# Full Python suite with writable task-owned paths — 2026-10-01

Status: `PASS` for the repository's 238-test Python suite.

The suite was run from the existing `app/server/.venv` after restoring the
locked `test` extra. The backend ran on `127.0.0.1:5503` and the built
frontend preview ran on `127.0.0.1:8503`. Browser screenshots and E2E evidence
were redirected to the sibling `full-suite-live-20261001-r5/` directory;
pytest basetemp and cache were under the task-owned
`runtimes/cache/release-gate-full-live-20261001-r5/` directory.

Command shape:

```powershell
app/server/.venv/Scripts/python.exe -m pytest `
  -c app/server/pyproject.toml app/tests -v --tb=short `
  --basetemp runtimes/cache/release-gate-full-live-20261001-r5/pytest-tmp `
  -o cache_dir=runtimes/cache/release-gate-full-live-20261001-r5/pytest
```

Result: `235 passed, 3 skipped, 0 failed, 0 errors` in `123.11s`.
The run included browser E2E, API E2E, integration, and unit tests. The
PostgreSQL cases were the three documented skips. Both live services and the
task-owned listeners were stopped after the run; this does not claim the
separate npm client test commands from `run_tests.bat`.
