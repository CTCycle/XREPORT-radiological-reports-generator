# Tauri Windows Packaging Playbook

Last updated: 2026-04-08

Operational playbook for building and validating XREPORT desktop releases on Windows.

## 1. Purpose

This flow keeps:
- local development simple (`XREPORT/start_on_windows.bat`)
- desktop release builds explicit (`release/tauri/build_with_tauri.bat`)
- user artifacts centralized under `release/windows`

## 2. Build Workflow

1. Activate desktop env profile:
   - verify `XREPORT\settings\.env` runtime values
2. If branding changed, regenerate icons:
   - `cd XREPORT\client && npm run tauri:icon`
3. Build desktop release:
   - `release\tauri\build_with_tauri.bat`
4. Collect outputs:
   - `release/windows/installers`
   - `release/windows/portable`

## 3. Script Responsibilities

- `XREPORT/start_on_windows.bat`
  - installs/updates portable runtimes under `runtimes`
  - syncs Python deps with `uv`
  - builds frontend for local runtime
- `release/tauri/build_with_tauri.bat`
  - validates required portable runtimes
  - stages short-path bundle sources under `XREPORT/client/src-tauri/r`
  - runs `npm run tauri:build:release`
- `npm run tauri:icon`
  - regenerates desktop icon outputs from `XREPORT/client/public/favicon.png`
- `npm run tauri:clean` / maintenance option 3
  - removes desktop build residue

## 4. Resource Scope Rules

`XREPORT/client/src-tauri/tauri.conf.json` uses an explicit `bundle.resources` map.
Keep this whitelist explicit; avoid broad recursive globs.

Current staged coverage includes:
- backend app folders (`server`, `scripts`, `settings`)
- frontend dist
- runtime templates/tokenizer assets
- portable runtimes (`python`, `uv`, `nodejs`)
- `pyproject.toml` and `uv.lock`

## 5. Pre-Release Verification

1. Build succeeds with `release\tauri\build_with_tauri.bat`.
2. Artifacts exist in `release/windows/installers`.
3. Installer/portable app starts on a clean machine.
4. Desktop UI can reach backend routes.
5. First-run experience is acceptable (including possible dependency sync delay).
6. Splash/status text does not expose absolute filesystem paths.
7. Expected icon appears for both installer and app executable.

## 6. Common Failure Modes

- Cargo missing:
  - install Rust toolchain via rustup on build machine.
- npm/node missing:
  - ensure portable runtime exists (run `XREPORT/start_on_windows.bat`).
- Tauri build failure:
  - inspect `npm run tauri:build:release` output in `XREPORT/client`.
- icon generation failure:
  - verify `XREPORT/client/public/favicon.png` is valid square PNG.
- stale icon shown in Explorer:
  - Explorer icon cache may lag; verify binary replacement and refresh cache/path.
- long splash startup:
  - expected on first run when runtime sync is needed for heavy ML dependencies.
