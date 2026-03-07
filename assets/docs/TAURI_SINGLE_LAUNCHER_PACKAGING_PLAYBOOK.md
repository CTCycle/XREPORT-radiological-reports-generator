# Tauri Single-Launcher Packaging Playbook (Windows)

## 1. Purpose

This playbook documents the exact packaging pathway used in XREPORT to provide a one-click desktop launcher with these behaviors:

- user clicks one script only (`start_on_windows_tauri.bat`)
- if desktop executable is missing, it builds on first run
- Tauri starts backend through the same script using an internal mode (`--backend`)
- no separate backend script is exposed to users
- closing the Tauri main window terminates backend process tree

Use this as a template for another project with the same requirements.

## 2. Target Runtime Flow

1. User runs `XPROJECT/start_on_windows_tauri.bat`
2. Script checks for existing built Tauri executable
3. If missing (or first run marker missing), script runs `npm run tauri:build`
4. Script launches desktop executable
5. Tauri Rust app starts backend using:
   - `cmd /c start_on_windows_tauri.bat --backend`
6. Backend mode in the same batch file:
   - ensures portable Python and uv
   - runs dependency sync
   - launches uvicorn foreground process
7. On Tauri exit, Rust code kills backend process tree (`taskkill /T /F`)

## 3. File Structure Pattern

Required files and roles:

- `XPROJECT/start_on_windows_tauri.bat`
  - public launcher mode (default)
  - internal backend mode (`--backend`)
- `XPROJECT/client/src-tauri/src/main.rs`
  - spawns backend with `--backend`
  - waits for backend port readiness
  - redirects webview to local backend URL
  - kills backend process tree on exit
- `XPROJECT/client/src-tauri/tauri.conf.json`
  - bundles project resources needed by backend bootstrap/runtime

Optional runtime assets:

- `XPROJECT/resources/runtimes/python`
- `XPROJECT/resources/runtimes/uv`

## 4. Batch Launcher Design

Implement two modes in the same script:

- mode A: user launcher (default, no args)
- mode B: backend bootstrap (`--backend`)

### 4.1 User launcher mode

Responsibilities:

- create `.env` from Tauri template if missing
- locate executable candidates:
  - `client/src-tauri/target/release/xproject-desktop.exe`
  - optional fallback names
- enforce one-time rebuild marker, for example:
  - `client/src-tauri/target/.xproject_tauri_single_launcher_v1`
- if build required:
  - verify `cargo` is available
  - verify `npm` (or project portable npm) is available
  - run `npm ci` (or `npm install` if lock missing)
  - run `npm run tauri:build`
  - write marker file
- launch executable and exit

### 4.2 Backend mode (`--backend`)

Responsibilities:

- ensure runtime directories exist
- ensure portable Python embeddable runtime exists
- ensure portable `uv` exists
- parse `.env` overrides (host, port, reload, optional deps)
- set runtime env:
  - `PYTHONHOME`
  - `PYTHONNOUSERSITE=1`
  - `XPROJECT_TAURI_MODE=true`
- run dependency sync (`uv sync`)
- run backend (`uv run python -m uvicorn ...`) in foreground

Important: backend mode should not spawn detached child processes itself. Let Tauri own the process chain.

## 5. Rust (Tauri) Integration Pattern

In `main.rs`:

1. Resolve launcher path candidates (resource dir + workspace fallbacks)
2. Spawn backend via:

```rust
Command::new("cmd")
    .arg("/c")
    .arg(launcher_path)
    .arg("--backend")
```

3. Poll TCP connect to `FASTAPI_HOST:FASTAPI_PORT` until ready
4. On ready, load `http://host:port/` in the main window
5. On `RunEvent::Exit` and `RunEvent::ExitRequested`, terminate process tree:

```rust
Command::new("taskkill")
    .args(["/PID", &child.id().to_string(), "/T", "/F"])
```

This prevents orphaned Python/uvicorn processes when the window closes.

## 6. Tauri Config Requirements

`tauri.conf.json` should include:

- app window with `about:blank` startup URL (or your preferred placeholder)
- bundled resources that backend needs at runtime
- frontend build hooks (`beforeBuildCommand`, `frontendDist`)

For release no-console behavior, keep in Rust:

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
```

## 7. Migration Checklist (Another Project)

1. Create a single `start_on_windows_tauri.bat` with dual mode (`--backend`)
2. Remove public backend-only script from root folder
3. Update `main.rs` to call single launcher with `--backend`
4. Add process-tree kill on Tauri exit
5. Ensure backend serves UI origin expected by Tauri
6. Ensure `tauri.conf.json` bundles required backend/runtime assets
7. Add first-run rebuild marker logic in launcher
8. Verify first-run on a clean Windows machine (no prebuilt exe)
9. Verify second run skips build and launches directly
10. Verify closing window kills backend process tree

## 8. Verification Protocol

Run these checks in order:

1. Delete `client/src-tauri/target/release/*.exe`
2. Run `start_on_windows_tauri.bat`
3. Confirm build occurs and desktop window opens
4. Confirm backend responds on configured local port
5. Close desktop window
6. Confirm no `python`, `uv`, or `uvicorn` process remains for the app
7. Run launcher again and confirm it starts quickly without rebuild

## 9. Common Failure Modes

- `cargo` missing
  - install Rust toolchain (`rustup`)
- `npm` missing
  - install Node.js or provide project-local portable Node/npm
- backend never becomes reachable
  - check `.env` host/port and firewall
  - check backend logs from launcher backend mode
- app closes but backend remains
  - ensure `taskkill /T /F` is executed on both exit events

## 10. Recommended Commit Scope

When porting this pathway, keep one commit for:

- unified launcher (dual mode)
- Tauri Rust backend spawn + exit termination updates
- removal of extra user-facing backend launcher
- docs update explaining the packaging flow
