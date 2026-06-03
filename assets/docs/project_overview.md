# XREPORT Project Overview

Last updated: 2026-06-03

## Purpose

This file is the root index for `assets/docs`. Read it first to find the narrowest topic file for the active question.

## How To Navigate

1. Start with this file only.
2. Identify the topic area that matches the task.
3. Open the smallest leaf file that covers the needed detail.
4. Open sibling files only when the task clearly crosses topic boundaries.
5. Do not read the entire tree unless the task explicitly requires broad context.

## Naming Rules

- All documentation files and folders under `assets/docs` use lowercase names.
- Root-level files are reserved for top-level entry points.
- Topic folders group narrower leaf files by subject so large markdown files do not need to be loaded by default.

## Documentation Ontology

### Root

- `project_overview.md`
  - Root index, navigation rules, and environment guidance.

### Architecture

- `architecture/system_overview.md`
  - Repository layout, runtime topology, and entry points.
- `architecture/backend_api.md`
  - Mounted routers and endpoint catalog.
- `architecture/execution_and_data_flow.md`
  - Layer responsibilities, job model, and async versus sync behavior.
- `architecture/persistence.md`
  - Database mode selection, initialization behavior, and artifact locations.

### Coding

- `coding/python.md`
  - Python runtime, typing, validation, concurrency, and structure rules.
- `coding/typescript.md`
  - Frontend structure, typing, API integration, and styling rules.
- `coding/testing_and_quality.md`
  - Tooling, testing expectations, Windows scripting rules, and documentation discipline.

### Runtime

- `runtime/modes.md`
  - Supported runtime modes and operational constraints.
- `runtime/startup.md`
  - Launcher, manual startup, desktop build, and test procedures.
- `runtime/configuration.md`
  - Shared configuration sources, environment variables, and interoperability.
- `runtime/deployment.md`
  - Packaging prerequisites, Windows distribution, and runtime lock notes.

### UI

- `ui/design_tokens.md`
  - Typography, spacing, layout, color, and token guidance.
- `ui/components_and_patterns.md`
  - Reusable component patterns, states, and route-level structure.
- `ui/experience.md`
  - UX flows, responsiveness, accessibility, and design principles.

### Operations

- `operations/getting_started.md`
  - Intended users, startup paths, and runtime entry options.
- `operations/workflows.md`
  - Core dataset, training, inference, and validation journeys.
- `operations/commands_and_locations.md`
  - Primary commands, best practices, features, and output locations.
- `operations/troubleshooting.md`
  - Quick troubleshooting and database initialization behavior.

## Reading Order

1. Read `project_overview.md`.
2. Open the smallest leaf file that answers the question.
3. Expand to adjacent files only when the task crosses topic boundaries.
4. Return here before jumping to a different topic branch.

## Context Rules

- Read documentation files only when required by the active task.
- Defer reading until the task proves the file is needed.
- Keep affected docs updated whenever implementation changes alter behavior.
- Always include a `Last updated: YYYY-MM-DD` line when modifying a document.
- Pre-select relevant docs from folder structure and user intent before opening more files.

## Environment Rules

- Assume Windows as the default operating environment for commands and paths.
- Document both CMD and PowerShell usage when commands differ.
- Prefer commands runnable from repository root with explicit relative paths.
- Keep environment guidance aligned with `XREPORT/start_on_windows.bat`, `XREPORT/setup_and_maintenance.bat`, `release/tauri/build_with_tauri.bat`, and `tests/run_tests.bat`.
