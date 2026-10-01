# Runtime Startup

Last updated: 2026-09-24

## Windows Local Launcher

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1
```

For direct, non-interactive launch use:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action Launch
```

To force-stop XREPORT source services, desktop development shells, and
packaged XREPORT processes, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action KillProcesses
```

This action also clears the configured backend and frontend listeners. It
does not close the browser window.

The menu can:

- prepare portable Python, uv, and Node.js in `runtimes/`
- synchronize backend and frontend dependencies
- rebuild the frontend without launching services
- build and launch the local web application
- initialize the database and run tests
- create or remove selected desktop release artifacts
- remove logs, clear caches, or uninstall generated dependencies
- update source from `origin/main` with the clean-`main` guard

The launch option starts FastAPI and a lightweight Node server for the existing
production Angular bundle, waits only for the UI port to respond, attempts to
open the browser, and then exits the menu. If Windows denies the automatic
browser launch, both services remain running and the launcher prints the UI URL
for manual opening. The Node server serves the bundle, applies
Angular SPA fallback, and proxies `/api` to FastAPI. The Angular shell displays
the XREPORT startup surface while it polls `/api/health`; routed pages are not
created until the backend reports `status: "ok"`.

Before dependency preparation or service startup, Launch checks both configured
ports. If listeners are found, the launcher lists each unique PID, process
metadata when available, and every configured port it owns. Interactive Launch
asks once before terminating the listed process trees. A declined prompt
cancels without starting services; non-interactive Launch fails closed without
terminating anything. A launcher or ancestor process owning a configured port,
failed termination, or a newly appearing owner aborts the launch.

For isolated startup validation, a process-level `XREPORT_RESOURCES_DIR`
override takes precedence over the same key in `settings/.env`. This lets a
disposable database be selected without editing the developer's environment
file or application resources.

## Foundational Startup Validation

Startup validation is the Tier 0 foundation of the campaign in
[`validation_campaign_ledger.md`](../validation_campaign_ledger.md):

- S01 uses a disposable `XREPORT_RESOURCES_DIR` to prove fresh SQLite startup,
  Alembic head readiness, required resource creation, restart reuse, and
  fail-closed rejection of a non-empty database without `alembic_version`.
- S02 opens the frontend before backend readiness and verifies that routed
  feature surfaces and model-catalogue requests remain suppressed until
  `/api/health` succeeds. The shell must represent ready, slow, unavailable,
  retry/recovery, and post-ready feature-error states without returning to the
  startup gate for an ordinary API error.

The acceptance boundary is rendered behavior plus backend evidence, not merely
an HTTP 200 from the preview server. Keep the slice summary and screenshots in
the tracked campaign evidence directory when the result is intended to be
reused by another checkout.

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

If the project virtual environment was created by an older pinned Python
patch release, the launcher recreates that disposable environment before
synchronizing the locked dependencies.

At backend startup, a missing `settings/.env` is created from
`settings/.env.example`. Existing environment files are preserved and ignored
by Git.

Normal Launch reuses a valid `app/client/dist/client-angular` bundle. After a
successful production build, the launcher writes the ignored
`.xreport-build-state.json` manifest beside that bundle. Its SHA-256 inputs are
the production Angular source, public assets, build configuration, dependency
manifests, and the expected Node toolchain version; test files, the dev proxy
configuration, backend files, documentation, and port-only environment edits
do not invalidate it. A missing or stale bundle rebuilds, and a dependency
manifest change runs `npm ci` before rebuilding. A current bundle never invokes
the Angular CLI during normal Launch. The explicit **Rebuild frontend** and
install/update actions always refresh the production bundle and state manifest.

### Source updates

**Update application from main** is menu option 6 and the direct
`-Action Update` command. It requires a non-detached, clean checkout of
`main` and runs `git pull --ff-only origin main`; it does not switch branches or
modify local changes. This source update is separate from dependency install or
frontend rebuild actions.

## Tauri desktop development

`LaunchDesktopDev` builds Angular once, starts the source FastAPI backend on
the configured 5003 port, starts the built-bundle server on 8003, waits only
for the frontend, leaves both consoles visible, and opens the debug Tauri shell. The
same Angular startup gate remains visible until backend readiness:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action LaunchDesktopDev
```

The existing `-Action Launch` path remains the normal browser workflow and is
not replaced by the desktop action.

## Tauri release packaging

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 `
  -Action BuildDesktopRelease -DesktopRuntime All -DesktopTarget All -Version 3.1.0
```

Use `-DesktopRuntime Cpu|Cuda`, `-DesktopTarget Portable|Msi`, `-Force`, and
`-OfflineWebView2` as needed. A clean tree is required unless
`-AllowDirtyTree` is explicitly supplied for diagnostic work. Outputs are
under `release/`; generated Cargo/PyInstaller/runtime staging is under
`app/desktop/build` and is ignored. Remove it with:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action RemoveDesktopRelease
```

The release action always recreates `app/desktop/src-tauri/ui` and
`app/desktop/src-tauri/generated/runtime.zip` from the locked frontend/backend
inputs. It uses separate Cargo targets for CPU and CUDA and fails closed if a
release Tauri build is invoked without a valid generated runtime. From
`app/desktop`, `npm run tauri:build` is the complete CPU release wrapper; use
`npm run tauri:build -- -DesktopRuntime Cuda` for the CUDA variant.

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
`app/desktop/build/cargo-target/<variant>/release/xreport-desktop.exe`; it is a shell build,
not a distributable portable artifact.

Release version input is canonicalized in `app/server/pyproject.toml`. The
launcher validates the client package, backend metadata, Cargo package, Tauri
CPU/CUDA configurations, and generated OpenAPI version against that value.

## Manual Backend And Frontend

Source-mode manual commands use the repository's canonical disposable-cache
root. From the repository root, set `XREPORT_CACHE_ROOT="$PWD/runtimes/cache"`
and the related `XDG_CACHE_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`,
`NPM_CONFIG_CACHE`, `PLAYWRIGHT_BROWSERS_PATH`, `PYTHONPYCACHEPREFIX`, and
`MPLCONFIGDIR` variables below that root before running these commands. The
Windows launcher exports the same locations automatically.

PowerShell:

```powershell
uv run --project app/server python -m uvicorn server.app:app --app-dir app --host 127.0.0.1 --port 5003
Set-Location app/client
npm run preview -- --host 127.0.0.1 --port 8003
```

Use host and port values from `settings/.env`. Run `npm run build` before the
first manual preview start. The `preview` script serves the built bundle and
proxies `UI_API_BASE_URL` (normally `/api`) to the FastAPI host and port from
the environment. Use `npm start` or `npm run dev` when an Angular development
server is required.

## Test Runtime

CMD:

```cmd
app\tests\run_tests.bat
```

The test launcher uses the prepared backend environment and starts required local services when they are not already running.

On backend startup, the shared Alembic coordinator creates the SQLite file or
PostgreSQL database when necessary and upgrades to the checked-in head before
the readiness callback runs. An existing database with application tables but
without migration state is rejected; startup never infers or stamps an
unversioned schema. The launcher’s explicit database option uses the same
coordinator. The `application_settings` singleton is created or migrated from
the one-time legacy import before readiness; after that, settings are read only
from the database. Startup creates required resource directories for logs,
models, tokenizers, checkpoints, and templates. The migration stream removes
obsolete report-job state and registers complete checkpoint artifacts in the
canonical database registry.

## Development Cache Locations

All disposable application, ML, frontend, runtime, and test caches are kept
under the single canonical `runtimes/cache` root. Its subdirectories include
`pytest`, `pytest-tmp`, `ruff`, `python`, `coverage`, `angular`, `uv`, `pip`,
`npm`, `playwright-browsers`, `huggingface`, `torch`, `keras`, and
`matplotlib`. Packaged launches use the same hierarchy below the writable
`<data-root>/runtimes/cache` path; persistent models and application data stay
outside it.

The cleanup action also performs a narrowly scoped, cleanup-only sweep for
legacy root pytest/Ruff/uv directories, the former `app/tests/cache` trees,
client-local cache directories, and old transient model-cache paths. Those
legacy paths are never active configuration and are not recreated.

Select **Clear cache** in the maintenance menu, or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action ClearCache
```

Cache cleanup is best-effort and reproducible: each target is enumerated and
its items are removed individually in deterministic deepest-first order. Locked
or administrator-protected files are reported and skipped; other cached
artifacts continue to be removed. `.gitkeep` sentinels remain preserved.

The maintenance menu and direct actions also expose `RemoveCheckpoints` and
`RemoveAllData`. They use the same interactive `[y/N]` confirmation as log,
cache, uninstall, and desktop-release removal actions. `RemoveAllData` clears
the configured resource root's database sidecars, checkpoints, models,
tokenizers, and logs while preserving application files, templates, and
`.gitkeep` sentinels; external databases are not modified.
