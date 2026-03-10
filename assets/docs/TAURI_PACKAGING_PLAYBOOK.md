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
2. If desktop branding changed, maintainer regenerates desktop icons:
   - `cd XREPORT\client && npm run tauri:icon`
3. Maintainer runs release build helper:
   - `release\tauri\build_with_tauri.bat`
4. Script installs frontend dependencies and runs:
   - `npm run tauri:build:release`
5. Artifacts are exported to:
   - `release/windows/installers` (preferred for users)
   - `release/windows/portable` (raw app executable)
6. Maintainer publishes installer artifacts (for example, GitHub Releases).
7. End user runs installer/`.exe` directly.

## 3. Script Roles

- `XREPORT/start_on_windows.bat`
  - local mode (v1) web launcher
  - portable runtime bootstrap for development
- `release/tauri/build_with_tauri.bat`
  - release build helper for desktop packaging
  - build-time checks for Cargo, npm, and node runtime availability
- `XREPORT/client/package.json` -> `npm run tauri:icon`
  - regenerates desktop icon assets from `XREPORT/client/public/favicon.png`
  - removes generated mobile icon folders so the repo stays desktop-only
- `XREPORT/setup_and_maintenance.bat`
  - maintenance menu including `3. Clean desktop build artifacts`

## 4. Cleanup Guidance

If the desktop app branding changes, regenerate icons before packaging:
- `cd XREPORT\client && npm run tauri:icon`

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

1. Run `release\tauri\build_with_tauri.bat`.
2. Confirm artifacts exist under `release/windows/installers`.
3. Install/run generated package on a clean Windows machine.
4. Confirm app starts and backend endpoints are reachable from desktop UI.
5. Confirm end-user flow works without globally installed Rust/Cargo.
6. Confirm splash/status text does not expose absolute filesystem paths.
7. Confirm the installer and portable `.exe` show the expected app icon.

## 7. Common Failure Modes

- `cargo` missing
  - install Rust toolchain (`rustup`) on build machine.
- `npm` missing
  - install Node.js or bootstrap project-local Node via `start_on_windows.bat`.
- `node` not recognized while running npm package install scripts
  - ensure `build_with_tauri.bat` resolves `node.exe` and exports its folder in `PATH` before `npm ci`/`npm install`.
- `tauri:build:release` fails
  - inspect `npm run tauri:build:release` logs in `XREPORT/client`.
- icon generation fails
  - verify `XREPORT/client/public/favicon.png` is a real square PNG, not a renamed JPEG or another format.
- rebuilt app still shows the old icon in Windows Explorer
  - verify the built `.exe` was replaced, then rename the file, move it to a new folder, or refresh Windows icon cache because Explorer can keep stale icon cache entries for the same path/name.
- portable/installed app remains on startup splash for a long time
  - desktop v2 first looks for an existing `.venv` in discovered valid workspaces; when one is available it is reused.
  - if no reusable `.venv` exists, runtime is created under a writable root (`<workspace>` when writable, otherwise `%LOCALAPPDATA%\com.xreport.desktop\runtime`).
  - first launch may still spend minutes on `uv sync --frozen` because `torch`/`torchvision` are large.
  - verify the app can write `<runtime-root>\.venv` and `<runtime-root>\.uv-cache`.
- app starts but cannot reach backend
  - verify `XREPORT/settings/.env` host/port values and firewall rules.
