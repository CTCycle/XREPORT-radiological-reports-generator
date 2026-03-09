# Tauri Windows Packaging Playbook

## 1. Purpose

This playbook documents the Windows packaging workflow used in XREPORT.

Goals:
- keep local web development simple (`start_on_windows.bat`)
- keep desktop release builds explicit (`build_with_tauri.bat`)
- ship user-friendly installers from a predictable root-level folder
- avoid requiring end users to install Rust/Cargo

## 2. Current Workflow

1. Maintainer prepares desktop env profile:
   - `copy /Y XREPORT\settings\.env.local.tauri.example XREPORT\settings\.env`
2. Maintainer runs release build helper:
   - `XREPORT\build_with_tauri.bat`
3. Script installs frontend dependencies and runs:
   - `npm run tauri:build:release`
4. Artifacts are exported to:
   - `release/windows/installers` (preferred for users)
   - `release/windows/portable` (raw app executable)
5. Maintainer publishes installer artifacts (for example, GitHub Releases).
6. End user runs installer/`.exe` directly.

## 3. Script Roles

- `XREPORT/start_on_windows.bat`
  - local mode (v1) web launcher
  - portable runtime bootstrap for development
- `XREPORT/build_with_tauri.bat`
  - release build helper for desktop packaging
  - build-time checks for Cargo, npm, and node runtime availability
- `XREPORT/setup_and_maintenance.bat`
  - maintenance menu including `3. Clean desktop build artifacts`

## 4. Cleanup Guidance

Use one of these methods to clean desktop build residue:
- `cd XREPORT\client && npm run tauri:clean`
- `XREPORT\setup_and_maintenance.bat` -> option `3. Clean desktop build artifacts`

Cleanup removes build outputs only:
- `XREPORT/client/src-tauri/target/release`
- `release/windows`

Do not delete `XREPORT/client/src-tauri`; it contains source code and Tauri configuration.

## 5. Bundle Resource Scope

`XREPORT/client/src-tauri/tauri.conf.json` should keep an explicit resource whitelist.
Avoid broad recursive globs that can accidentally include generated build trees.

Current resource coverage includes:
- backend app folders (`server`, `scripts`, `settings`)
- runtime assets and templates used by local mode v2
- launcher scripts and lock/config files required at runtime

## 6. Verification Protocol

Run these checks before publishing:

1. Run `XREPORT\build_with_tauri.bat`.
2. Confirm artifacts exist under `release/windows/installers`.
3. Install/run generated package on a clean Windows machine.
4. Confirm app starts and backend endpoints are reachable from desktop UI.
5. Confirm end-user flow works without globally installed Rust/Cargo.

## 7. Common Failure Modes

- `cargo` missing
  - install Rust toolchain (`rustup`) on build machine.
- `npm` missing
  - install Node.js or bootstrap project-local Node via `start_on_windows.bat`.
- `node` not recognized while running npm package install scripts
  - ensure `build_with_tauri.bat` resolves `node.exe` and exports its folder in `PATH` before `npm ci`/`npm install`.
- `tauri:build:release` fails
  - inspect `npm run tauri:build:release` logs in `XREPORT/client`.
- app starts but cannot reach backend
  - verify `XREPORT/settings/.env` host/port values and firewall rules.

