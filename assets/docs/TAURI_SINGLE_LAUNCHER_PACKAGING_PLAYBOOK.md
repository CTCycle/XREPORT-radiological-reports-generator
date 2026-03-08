# Tauri Windows Packaging Playbook

## 1. Purpose

This playbook documents the current Windows packaging pathway used in XREPORT.

Goals:
- keep local web development simple (`start_on_windows.bat`)
- keep desktop release builds explicit (`build_with_tauri.bat`)
- ship pre-built installer/executable artifacts to end users
- avoid requiring end users to install Rust/Cargo

## 2. Current Workflow

1. Maintainer prepares desktop env profile:
   - `copy /Y XREPORT\settings\.env.local.tauri.example XREPORT\settings\.env`
2. Maintainer runs release build helper:
   - `XREPORT\build_with_tauri.bat`
3. Script installs frontend dependencies and runs:
   - `npm run tauri:build`
4. Artifacts are collected from:
   - `XREPORT/client/src-tauri/target/release/bundle`
5. Maintainer publishes installer/`.exe` artifacts (for example, GitHub Releases).
6. End user downloads and runs installer/`.exe` directly.

## 3. Script Roles

- `XREPORT/start_on_windows.bat`
  - local mode (v1) web launcher
  - portable runtime bootstrap for development
- `XREPORT/build_with_tauri.bat`
  - release build helper for desktop packaging
  - build-time checks for Cargo and npm

## 4. Production Guidance

- Build desktop artifacts on controlled build machines or CI runners.
- Sign installers/executables for distribution trust.
- Publish checksums with release artifacts.
- Treat batch scripts as maintainer tooling, not end-user entrypoints.

## 5. Verification Protocol

Run these checks before publishing:

1. Run `XREPORT\build_with_tauri.bat`.
2. Confirm artifacts exist under `client/src-tauri/target/release/bundle`.
3. Install/run generated package on a clean Windows machine.
4. Confirm app starts and backend endpoints are reachable from desktop UI.
5. Confirm end user flow works without globally installed Rust/Cargo.

## 6. Common Failure Modes

- `cargo` missing
  - install Rust toolchain (`rustup`) on build machine.
- `npm` missing
  - install Node.js or bootstrap project-local Node via `start_on_windows.bat`.
- `tauri:build` fails
  - inspect `npm run tauri:build` logs in `XREPORT/client`.
- app starts but cannot reach backend
  - verify `XREPORT/settings/.env` host/port values and firewall rules.

