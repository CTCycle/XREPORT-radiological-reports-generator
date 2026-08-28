You are working on the XREPORT-radiological-reports-generator repository checked out at the current directory.

TASK: Remove all Tauri packaging infrastructure, consolidate the launcher scripts into a single PowerShell menu, and update all documentation. Do NOT modify any Python, TypeScript, or Angular source code.

## Step 1: Create `app.ps1` at repo root

Replace both `start_on_windows.bat` and `setup_and_maintenance.bat` with a single `app.ps1` interactive menu.

Menu title: "XREPORT — Radiological Reports Generator"

The menu options and logic are identical to the description in PROMPT 1 (DILIGENT) Step 1. Read the existing start_on_windows.bat and setup_and_maintenance.bat in this repo for the exact paths, port defaults, and environment variable names used. Replicate their logic faithfully in PowerShell.

Key things to extract from the existing batch files:
- Python version, URL for embeddable download
- Node.js version, URL
- uv download URL  
- Default FASTAPI_HOST/FASTAPI_PORT and UI_HOST/UI_PORT
- uv sync working directory
- npm commands and working directory
- Backend uvicorn module path
- Health check endpoint

## Step 2: Delete old batch files

Remove from repo root:
- start_on_windows.bat
- setup_and_maintenance.bat

## Step 3: Delete all Tauri / Cargo / Rust artifacts

Directories to delete (entire trees):
- app/src-tauri/
- release/tauri/
- release/windows/ (if exists)

Files to delete:
- .github/workflows/desktop-release.yml

## Step 4: Update .gitignore

Remove entries for Tauri build outputs.

## Step 5: Update package.json (app/client/)

- Remove "@tauri-apps/cli" from devDependencies if present
- Remove any "build:tauri" script
- Keep the regular "build" script

## Step 6: Update README.md

Read the current README.md and make these changes:
- Remove references to "Local mode (v2)" and "packaged Windows desktop application built with Tauri"
- Remove the "Windows (Packaged Desktop, Local mode v2)" subsection
- Remove the entire desktop packaging section and its bullet points about src-tauri
- Remove "release\tauri\build_with_tauri.bat" and "release\windows\installers|portable" references
- Update Quick Start to reference app.ps1
- Change "start_on_windows.bat" references to "app.ps1"

## Step 7: Update assets/docs/

Scan for Tauri/desktop/packaging references and update. Pay special attention to:
- assets/docs/runtime/
- assets/docs/architecture/
- assets/docs/operations/

## Step 8: Verify

Check: batch files deleted, app/src-tauri/ gone, release/tauri/ gone, desktop-release.yml deleted, app.ps1 exists, no Tauri references in docs.