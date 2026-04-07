# General Rules for This Repository

This document is the primary source of truth for how work should be executed and documented in this repository.

## 1. Mandatory Read Order

Before starting any task:
1. Read `assets/docs/GENERAL_RULES.md`.
2. Read only the additional docs that are relevant to the task scope.

Use this mapping:
- `assets/docs/ARCHITECTURE.md`: system structure, APIs, runtime behavior.
- `assets/docs/BACKGROUND_JOBS.md`: job lifecycle, polling, cancellation.
- `assets/docs/GUIDELINES_PYTHON.md`: Python/FastAPI backend changes.
- `assets/docs/GUIDELINES_TYPESCRIPT.md`: React/TypeScript frontend changes.
- `assets/docs/GUIDELINES_TESTS.md`: test layout, runners, and test-writing rules.
- `assets/docs/PACKAGING_AND_RUNTIME_MODES.md`: env profiles, launch/build modes.
- `assets/docs/TAURI_PACKAGING_PLAYBOOK.md`: Windows desktop packaging flow.
- `assets/docs/README_WRITING.md`: README authoring and structure.

## 2. Source of Truth Priority

When docs disagree, resolve with this priority:
1. Current source code and scripts in the repository.
2. This file (`GENERAL_RULES.md`).
3. Task-specific docs in `assets/docs`.

If you discover drift, update the affected docs in the same change.

## 3. Core Engineering Principles

- Keep changes scoped to the user task; avoid broad refactors.
- Prefer small, verifiable increments: implement, wire, validate.
- Preserve existing project conventions for structure, naming, and architecture.
- Use PowerShell for commands in this repository (use `cmd /c` only when needed for `.bat`/CMD-specific behavior).

## 4. Runtime and Environment Rules

- Active runtime env file: `XREPORT/settings/.env`.
- No profile templates are maintained; edit `XREPORT/settings/.env` directly for runtime values.
- Python target version: `>=3.14` (`pyproject.toml`).
- If `runtimes/.venv` exists, use it for Python commands and tests.

## 5. Documentation Maintenance Rules

Update docs when changes affect behavior, architecture, workflows, or operations.

Minimum expectations for every doc update:
- Commands are runnable as written.
- File paths and endpoint paths exist.
- Terminology is consistent across all docs.
- Examples match current runtime mode names:
  - Local mode (v1): web launcher flow.
  - Local mode (v2): packaged Tauri desktop flow.

## 6. Skills and External References

- Use relevant skills from the skills repository when the task benefits from reusable workflows.
- Use web verification when facts are likely to change or precision is required.

