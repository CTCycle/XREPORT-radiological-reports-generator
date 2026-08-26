# Runtime Startup

Last updated: 2026-08-21

## Windows Local Launcher

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1
```

For direct, non-interactive launch use:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action Launch
```

The menu can:

- prepare portable Python, uv, and Node.js in `runtimes/`
- synchronize backend and frontend dependencies
- rebuild the frontend without launching services
- build and launch the local web application
- initialize the database and run tests
- create or remove selected desktop release artifacts
- remove logs, clear caches, or uninstall generated dependencies

The launch option starts the backend, waits for `/api/health`, starts the frontend preview, waits for the UI port to respond, opens the browser, and then exits the menu.

Choose **Rebuild frontend only** to prepare the portable Node.js runtime and
frontend dependencies as needed, rebuild the Angular client, and leave backend
services untouched. The same operation can be run directly with:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action RebuildFrontend
```

The install/update option prepares the portable runtimes before requesting an
installation profile:

- `Development` includes Ruff, Pyright, and pytest.
- `Standard` installs runtime dependencies only.

At backend startup, a missing `settings/.env` is created from
`settings/.env.example`. Existing environment files are preserved and ignored
by Git.

Set `ALWAYS_REBUILD=true` in `settings/.env` to rebuild the frontend during
application launch. The default `ALWAYS_REBUILD=false` skips that startup
build; the install/update option continues to build the frontend.

## Tauri desktop development

`LaunchDesktopDev` builds Angular once, starts the source FastAPI backend on
the configured 5003 port, starts the preview on 8003, leaves both consoles
visible, and opens the debug Tauri shell:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action LaunchDesktopDev
```

The existing `-Action Launch` path remains the normal browser workflow and is
not replaced by the desktop action.

## Tauri release packaging

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 `
  -Action BuildDesktopRelease -DesktopRuntime All -DesktopTarget All -Version 3.0.0
```

Use `-DesktopRuntime Cpu|Cuda`, `-DesktopTarget Portable|Msi`, `-Force`, and
`-OfflineWebView2` as needed. A clean tree is required unless
`-AllowDirtyTree` is explicitly supplied for diagnostic work. Outputs are
under `release/`; generated Cargo/PyInstaller/runtime staging is under
`app/desktop/build` and is ignored. Remove it with:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action RemoveDesktopRelease
```

The interactive launcher has a **DESKTOP RELEASE** section. **Create release
artifacts** prompts for a version and then offers CPU portable, CPU MSI, CUDA
portable, CUDA MSI, or all four packages. **Remove release artifacts** offers
the same choices for that version and keeps the remaining variant checksum and
build metadata sidecars synchronized. These menu actions operate on final
`release/` payloads; the direct `RemoveDesktopRelease` action remains the
full cleanup for release, staging, PyInstaller, CPU overlay, Tauri target, and
generated runtime output.

Portable output is one EXE per variant: the release script streams the audited
ZIP64 runtime onto the PE as an overlay and writes a fixed footer. MSI output
keeps that ZIP as an installer resource. Do not copy only the raw
`app/desktop/src-tauri/target/release/xreport-desktop.exe`; it is a shell build,
not a distributable portable artifact.

## Manual Backend And Frontend

PowerShell:

```powershell
uv run --project app/server python -m uvicorn server.app:app --app-dir app --host 127.0.0.1 --port 5003
Set-Location app/client
npm run preview -- --host 127.0.0.1 --port 8003
```

Use host and port values from `settings/.env`. `UI_API_BASE_URL` should remain `/api` for the proxied local flow.

## Test Runtime

CMD:

```cmd
app\tests\run_tests.bat
```

The test launcher uses the prepared backend environment and starts required local services when they are not already running.

On backend startup, the shared Alembic coordinator creates the SQLite file or
PostgreSQL database when necessary, adopts only an exact known unversioned
schema, and upgrades to the checked-in head before the readiness callback runs.
The launcher’s explicit database option uses the same coordinator. Startup also
verifies the tracked configuration file and creates required resource
directories for logs, models, tokenizers, checkpoints, and templates.

## Development Cache Locations

Disposable runtime caches are kept under `runtimes/cache`, with separate
subdirectories for uv, npm, pip, and Playwright browser downloads. Pytest,
Ruff, Python bytecode, coverage, and other development-tool caches are kept
under `app/tests/cache`. The test runner uses
`app/tests/cache/pytest-tmp` for pytest's temporary test directory.

Select **Clear cache** in the maintenance menu, or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action ClearCache
```

Cache cleanup is best-effort: locked or administrator-protected files are
reported and skipped so the launcher can continue removing the remaining
artifacts.
