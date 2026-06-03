# Python Rules

Last updated: 2026-06-03

Apply these rules to new and modified Python code in scope of the task.

## Runtime And Dependencies

- Target Python version: `>=3.14` from `pyproject.toml`
- Use `runtimes/.venv` when it exists, otherwise follow project launcher and runtime conventions.
- Keep dependency state aligned with `uv` and `runtimes/uv.lock`.
- Do not create parallel or ad hoc environments for regular project work.

## Typing

- Type annotations are required for public APIs, endpoint methods, service methods, and non-trivial logic.
- Use built-in generics such as `list[str]`, `dict[str, Any]`, and `tuple[int, str]`.
- Prefer `|` unions over `typing.Union`.
- Use `collections.abc` for abstract collection types where appropriate.
- Treat typing as required quality, not optional documentation.

## Validation And APIs

- Use Pydantic and domain models for request and response validation.
- Avoid manual ad hoc validation when domain models can express the same constraints.
- Return explicit HTTP status codes consistently.
- Keep response contracts stable and explicit with domain response models.
- Ensure safe error handling and traceable job and request flows.

## Async And Concurrency

- Use `async` only when dependencies and call paths are non-blocking or require async I/O.
- Do not run CPU-heavy work directly inside async request handlers.
- Use the job system for long-running tasks through start, poll, and cancel endpoints.
- Keep cancellation cooperative and explicit in long-running workflows.

## Code Structure

- Keep functions focused and reasonably small.
- Make side effects explicit, especially for I/O, filesystem work, and database mutations.
- Prefer composable and straightforward control flow over deep nesting.
- Add comments only when they clarify safety or intent.
- Keep imports at the top of files unless a delayed import is technically required.
- Avoid nested function definitions unless strictly needed.
- Use classes to group related logic when the project already follows class-based patterns.
- Avoid broad style-only rewrites in unrelated code.
- Keep modules approximately under 1000 lines when practical.
