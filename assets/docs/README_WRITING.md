# README Writing Guidelines

Use this document when creating or updating `README.md` for this repository.

## 1. Goal

A README must help users run and use XREPORT without reading source code.
It should describe:
- what the app does
- how to install/run it
- how to use core workflows
- where outputs/configuration live

## 2. Required Structure

If a section does not apply, omit it and keep numbering contiguous.

1. Project Overview
2. Model and Dataset (optional, for ML context)
3. Installation
4. How to Use
5. Setup and Maintenance
6. Resources
7. Configuration
8. License

## 3. Content Rules

- Write for users, not internal implementation details.
- Keep instructions reproducible and command-accurate.
- Reflect current runtime mode names consistently:
  - Local mode (v1): web launcher
  - Local mode (v2): packaged desktop
- Use real paths and command examples from this repository.
- Avoid speculative claims; call out uncertainty explicitly.

## 4. Installation Section Rules

### 4.1 Windows
- Include `XREPORT/start_on_windows.bat` flow for local mode (v1).
- Include desktop packaging flow only for maintainer/build context.

### 4.2 macOS/Linux
- Include manual setup only if verified and supported by repository scripts/config.

## 5. How to Use Section Rules

- Describe operational workflow (dataset -> training -> inference -> validation).
- Include UI endpoints and launch behavior at a user level.
- If screenshots are used, keep file references valid and concise.

## 6. Maintenance and Resources

- Document `XREPORT/setup_and_maintenance.bat` actions at outcome level.
- Explain `XREPORT/resources` and `runtimes` responsibilities accurately.

## 7. Configuration Section Rules

- Reference active env file: `XREPORT/settings/.env`.
- Mention profile templates:
  - `XREPORT/settings/.env.local.example`
  - `XREPORT/settings/.env.local.tauri.example`
- Include a variable table only for variables that are actually used.

## 8. Cross-Reference Requirement

When README behavior changes materially, verify consistency with:
- `assets/docs/ARCHITECTURE.md`
- `assets/docs/PACKAGING_AND_RUNTIME_MODES.md`
- `assets/docs/GUIDELINES_TESTS.md`

