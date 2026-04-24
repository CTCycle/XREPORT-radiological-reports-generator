# XREPORT Project Overview

Last updated: 2026-04-24

## FILES INDEX

- ARCHITECTURE.md  
  System structure, entry points, API map, layer responsibilities, persistence model, and async/sync constraints.

- CODING_RULES.md  
  Language-specific coding standards (Python mandatory baseline + TypeScript/frontend standards), typing requirements, async policy, and tooling expectations.

- PROJECT_OVERVIEW.md  
  Documentation map, context-handling rules, and Windows environment guidance.

- RUNTIME_MODES.md  
  Supported runtime targets, startup commands, environment/config differences, interoperability, constraints, and deployment notes.

- UI_STANDARDS.md  
  UI design system and UX conventions based on the current React implementation (tokens, components, responsiveness, accessibility).

- USER_MANUAL.md  
  End-user operating guide for dataset, training, inference, validation, and troubleshooting workflows.

## CONTEXT RULES

- Read documents only when they are needed for the active task.
- Defer reading until the task requires specific context.
- Keep affected docs updated whenever implementation changes alter behavior.
- Always include a `Last updated: YYYY-MM-DD` line when modifying a document.
- Do not read all `SKILL.md` files indiscriminately.
- Pre-select relevant files from folder structure and user intent before opening them.

## ENVIRONMENT RULES

- Assume Windows as the default operating environment for commands and paths.
- Provide command equivalents when both CMD and PowerShell are relevant:
  - CMD for `.bat` workflows and legacy shell scripts.
  - PowerShell for scripting, file inspection, and automation tasks.
- Prefer commands that are directly runnable from repository root with explicit relative paths.
- Update this section when new Windows-specific operational patterns are introduced by the codebase.
