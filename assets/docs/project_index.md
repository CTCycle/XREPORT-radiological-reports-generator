# XREPORT Project Overview

Last updated: 2026-09-20

This file is the root index for `assets/docs`. Read it first to find the narrowest topic file for the active question.

## How To Navigate

1. Start with this file only.
2. Read `project_status_ledger.md` before substantial implementation or validation work to understand current operational state, active issues, and validation debt.
3. Identify the topic area that matches the task.
4. Open the smallest leaf file that covers the needed detail.
5. Open sibling files only when the task clearly crosses topic boundaries.
6. Do not read the entire tree unless the task explicitly requires broad context.

## Naming Rules

- All documentation files and folders under `assets/docs` use lowercase names.
- Root-level files are reserved for top-level entry points.
- Topic folders group narrower leaf files by subject so large markdown files do not need to be loaded by default.

## Documentation Ontology

### Root

- `project_index.md`
  - Documentation index, navigation rules, and environment guidance.
- `project_status_ledger.md`
  - Canonical current operational status, evidence summary, open issues, resolved findings, and validation debt.

### Architecture

- `architecture/system_overview.md`
  - Repository layout, runtime topology, dependency direction, and entry points.
- `architecture/backend_api.md`
  - Mounted routers, endpoint catalog, frontend client mapping, and root serving behavior.
- `architecture/execution_and_data_flow.md`
  - Domain, service, repository, provider, job model, service composition, and async versus sync behavior.
- `architecture/persistence.md`
  - Database mode selection, schema compatibility checks, entity ownership, constraints, and artifact locations.
- `architecture/architecture_review.md`
  - Findings, implementation status, deferred migration work, and validation evidence from the architecture review.

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
  - Windows launcher, manual startup, and test procedures.
- `runtime/configuration.md`
  - Shared configuration sources, environment variables, inference controls, and interoperability.
- `runtime/deployment.md`
  - Local deployment scope, Tauri release layout, and runtime preparation notes.
- `runtime/local_inference_models.md`
  - Embedded Hugging Face model catalogue, XREPORT checkpoints, and research-use constraints.

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

1. Read `project_index.md`.
2. Read `project_status_ledger.md` when the task involves implementation, debugging, release, or validation.
3. Open the smallest leaf file that answers the question.
4. Expand to adjacent files only when the task crosses topic boundaries.
5. Return here before jumping to a different topic branch.

## Context Rules

- Read documentation files only when required by the active task.
- Defer reading until the task proves the file is needed.
- Keep affected docs updated whenever implementation changes alter behavior.
- Keep `project_status_ledger.md` synchronized after implementation, regression discovery, meaningful validation, blocker changes, or issue resolution and revalidation.
- Treat the ledger as the canonical current-state summary; keep architecture documents authoritative for structure and contracts, validation reports authoritative for detailed evidence, and implementation plans authoritative for intended work.
- Always include a `Last updated: YYYY-MM-DD` line when modifying a document.
- Pre-select relevant docs from folder structure and user intent before opening more files.

## Environment Rules

- Assume Windows as the default operating environment for commands and paths.
- Document both CMD and PowerShell usage when commands differ.
- Prefer commands runnable from repository root with explicit relative paths.
- Keep environment guidance aligned with `start_on_windows.ps1` and `app/tests/run_tests.bat`.
