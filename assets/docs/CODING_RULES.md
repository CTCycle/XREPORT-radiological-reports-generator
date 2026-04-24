# Coding Rules

Last updated: 2026-04-24

## 1. Scope

These rules define enforceable coding standards for languages used in this repository. Apply them to new code and modified code in scope of the task.

## 2. Python Rules (Mandatory Baseline)

### Runtime and dependencies

- Target Python version: `>=3.14` (from `pyproject.toml`).
- Use `runtimes/.venv` when it exists; otherwise follow project launcher/runtime conventions.
- Keep dependency state aligned with `uv` and `runtimes/uv.lock`.
- Do not create parallel or ad-hoc environments for regular project work.

### Typing

- Type annotations are required for:
  - public APIs
  - endpoint/service methods
  - non-trivial logic
- Use built-in generics: `list[str]`, `dict[str, Any]`, `tuple[int, str]`.
- Prefer `|` unions over `typing.Union`.
- Use `collections.abc` for abstract collection types where appropriate.
- Treat typing as required quality, not optional documentation.

### Validation and APIs

- Use Pydantic/domain models for request/response validation and shape guarantees.
- Avoid manual ad-hoc validation when domain models can express the same constraints.
- Return explicit HTTP status codes (`200`, `202`, etc.) consistently.
- Keep response contracts stable and explicit with domain response models.
- Ensure safe error handling and traceable job/request flows.

### Async and concurrency

- Use `async` only when dependencies and call paths are non-blocking or require async I/O.
- Do not run CPU-heavy work directly inside async request handlers.
- Use the job system for long-running tasks (`start -> poll -> cancel` endpoints).
- Keep cancellation cooperative and explicit in long-running workflows.

### Code structure

- Keep functions focused and reasonably small.
- Make side effects explicit (I/O, filesystem, DB mutations).
- Prefer composable and straightforward control flow over deep nesting.
- Add comments only when needed for safety/intent clarity.
- Keep imports at the top of files unless a delayed import is technically required.
- Avoid nested function definitions unless strictly needed.
- Use classes to group related logic when the project already follows class-based patterns.
- Avoid broad style-only rewrites in unrelated code.
- Keep modules approximately under 1000 LOC when practical.

### Tooling and quality gates

- Linting/formatting: Ruff (or established project-equivalent toolchain).
- Type checking/editor baseline: Pylance-compatible typing quality.
- Testing: pytest, with coverage focused on `tests/unit` and impacted `tests/e2e`.
- Validate changed API contracts and job-lifecycle behavior with targeted tests.

## 3. TypeScript / Frontend Rules

### Baseline

- Stack: React 18 + TypeScript + Vite.
- Keep TypeScript strictness aligned with existing `tsconfig` settings.
- Preserve the existing structure:
  - `src/pages`
  - `src/components`
  - `src/services`
  - `src/hooks`
  - `src/types`

### Typing and state

- Avoid `any` unless unavoidable and narrowly scoped.
- Model API payloads and job contracts with explicit types/interfaces.
- Keep route-level state local when possible; use shared context/hooks only when cross-page state is required.

### API integration

- Route API calls through `src/services/*` modules.
- Keep `/api` contract alignment with backend endpoints and response models.
- Keep long-running UX in the established start/poll/cancel pattern.

### Styling and components

- Reuse existing tokens and CSS variables from `src/index.css`.
- Keep component CSS colocated by feature/component where already established.
- Preserve keyboard focus visibility and disabled/loading states for interactive controls.

## 4. Scripts and Shell Rules (Windows)

- Maintain compatibility with existing CMD launcher/build scripts (`.bat`) where those scripts are the operational entrypoints.
- Use PowerShell for advanced scripting/automation when needed.
- Keep environment variable names and path semantics consistent with current scripts.

## 5. Documentation and Change Discipline

- When behavior/contracts/runtime modes change, update corresponding docs in `assets/docs` in the same change.
- Keep changes scoped; do not refactor unrelated modules unless explicitly required.
