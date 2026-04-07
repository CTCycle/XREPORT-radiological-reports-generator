# Python Guidelines (XREPORT)

Repository-scoped standards for Python backend and scripts.

## 1. Runtime and Environment

- Target Python version is `>=3.14` (`pyproject.toml`).
- Prefer the existing virtual environment at `runtimes/.venv` when present.
- Runtime env file is `XREPORT/settings/.env`.

## 2. Project Structure

Use existing backend layering:
- `XREPORT/server/api`: FastAPI route modules
- `XREPORT/server/domain`: Pydantic/domain models
- `XREPORT/server/services`: business services and job logic
- `XREPORT/server/repositories`: DB backends, queries, serializers
- `XREPORT/server/learning`: ML training/inference logic

Do not move logic across layers unless required by the task.

## 3. Typing and Signatures

- Keep type hints on public functions, endpoint handlers, and non-trivial internals.
- Prefer built-in generics (`list`, `dict`, `tuple`) and `|` unions.
- Keep response/request models explicit through Pydantic models in `domain`.

## 4. FastAPI Conventions

- Follow class-based endpoint modules that register routes with `add_api_route`.
- Use `HTTPException` with precise status codes and actionable messages.
- For long-running tasks, use `job_manager`-based start/poll/cancel endpoints.
- Keep endpoint handlers focused on orchestration; place heavy logic in services.

## 5. Data and Persistence

- Use repository/serializer abstractions (`DataSerializer`, `ModelSerializer`, DB backends) for persistence behavior.
- Do not bypass DB abstractions with ad-hoc direct SQL in route modules.
- Keep SQLite/PostgreSQL mode behavior compatible with `database.embedded_database` in `XREPORT/settings/configurations.json`.

## 6. Imports, Paths, and Logging

- Keep imports at module top when practical.
- If a delayed import is necessary (for startup cost or cycle avoidance), keep it local and brief.
- Use `os`/`os.path` patterns already used in the repository for path handling consistency.
- Use project logger utilities where available instead of raw `print` for backend/runtime logs.

## 7. Style and Quality

- Preserve existing naming and module structure unless task scope requires change.
- Avoid broad refactors during feature/bug fixes.
- Keep functions cohesive and side effects explicit.

Preferred quality tooling:
- Formatter/linting: Ruff (or existing formatter choices in the current branch)
- Type checking: mypy when configured for the changed scope
- Tests: pytest

## 8. Testing Expectations

- Add or update tests when behavior changes.
- Keep tests deterministic and isolated.
- For job-based flows, assert API contract shape and terminal job outcomes rather than timing-sensitive internals.

See `assets/docs/GUIDELINES_TESTS.md` for test layout and execution commands.

