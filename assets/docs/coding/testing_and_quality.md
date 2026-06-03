# Testing And Quality Rules

Last updated: 2026-06-03

## Tooling And Quality Gates

- Linting and formatting use Ruff or the established project-equivalent toolchain.
- Typing should remain Pylance-compatible.
- Testing uses pytest with coverage focused on `tests/unit` plus impacted `tests/e2e`.
- Validate changed API contracts and job-lifecycle behavior with targeted tests.

## Windows Script Rules

- Maintain compatibility with existing CMD launcher and build scripts where those scripts are the operational entrypoints.
- Use PowerShell for advanced scripting and automation when needed.
- Keep environment variable names and path semantics consistent with current scripts.

## Documentation And Change Discipline

- When behavior, contracts, or runtime modes change, update the corresponding docs in `assets/docs` in the same change.
- Keep changes scoped. Do not refactor unrelated modules unless explicitly required.
