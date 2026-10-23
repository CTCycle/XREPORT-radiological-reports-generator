# Runtime Startup

Last updated: 2026-10-21

## Windows Local Launcher

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1
```

For direct, non-interactive launch use:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action Launch
```

To stop identifiable XREPORT source services, desktop development shells, and
packaged XREPORT processes, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action KillProcesses
```

This action leaves processes on configured ports alone unless their command
line or executable identifies them as XREPORT. Unknown listeners are reported
and left running. It does not close the browser window.

The menu can:

- prepare portable Python, uv, and Node.js in `runtimes/`
- synchronize backend and frontend dependencies
- rebuild the frontend without launching services
- build and launch the local web application
- initialize the database and run tests
- create or remove selected desktop release artifacts
- remove logs, clear caches, or uninstall generated dependencies
- update source from `origin/main` with the clean-`main` guard

Normal Windows Launch sets the private `XREPORT_SERVE_CLIENT` source-mode flag
and starts only FastAPI. FastAPI serves the built Angular bundle from
`app/client/dist/client-angular/browser`, falls back to `index.html` for
extensionless Angular routes, and keeps `/api` requests in the API namespace.
The browser and API use one origin at `http://FASTAPI_HOST:FASTAPI_PORT`; a
wildcard bind address opens through loopback. The backend root remains a `/docs`
redirect when the launcher flag is off. The Angular shell polls `/api/health`
and can open the restricted Database recovery screen when health reports
`database_recovery`.

Launch waits for `/api/health`, opens the application, then keeps this launcher
console as the lifecycle owner. Press Ctrl+C to stop. The launcher requests
FastAPI's authenticated shutdown endpoint and force-stops only the backend
process tree it started if graceful shutdown does not finish. Backend logs stay
in this console. If automatic browser launch fails, use the printed URL.

Normal Launch checks only `FASTAPI_PORT`; `LaunchDesktopDev` checks both
`FASTAPI_PORT` and `UI_PORT`. A port owner is stopped only when its process is
identified as XREPORT, with confirmation during interactive launch. Unknown
owners are reported and left running, and the launch stops with an error rather
than terminating them. `KillProcesses` follows the same ownership rule.

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
an HTTP 200 from a server process. Save captures under `assets/QA/`, then
record the result, source revision, scope, and remaining limits in the existing
campaign or status ledger. Do not add a separate summary file for each slice.

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

`LaunchDesktopDev` builds Angular once, starts source FastAPI and the built-
bundle server as supervised child processes, waits for backend and frontend
readiness, then runs Tauri in the launcher console. The backend and bundle
server write timestamped stdout/stderr logs below the configured resource
root's `logs/` directory. Closing the Tauri shell or pressing Ctrl+C requests
backend shutdown and stops only those child process trees:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action LaunchDesktopDev
```

The existing `-Action Launch` path remains the normal browser workflow and is
not replaced by the desktop action.

## Tauri release packaging

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 `
  -Action BuildDesktopRelease -DesktopRuntime All -DesktopTarget All -Version 3.3.1
```

Use `-DesktopRuntime Cpu|Cuda`, `-DesktopTarget Portable|Msi`, `-Force`, and
`-OfflineWebView2` as needed. A clean tree is required unless
`-AllowDirtyTree` is explicitly supplied for diagnostic work. Outputs are
under `release/`; generated Cargo/PyInstaller/runtime staging is under
`app/desktop/build` and is ignored. To remove every desktop release and its
generated build state, choose **All versions and artifacts** from the
launcherâ€™s **Remove release artifacts** menu or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action RemoveDesktopRelease -AllDesktopReleases
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
the same package choices, **All artifacts for one version**, and **All versions
and artifacts**. Removing selected packages for one version keeps the
remaining variant checksum and build metadata sidecars synchronized. **All
versions and artifacts** removes every desktop release plus generated desktop
build state, including Cargo targets, PyInstaller/runtime staging, desktop
dependencies, and generated Tauri files. It preserves tracked source and build
scripts, `assets/QA/` evidence, installed applications, and user data. The
`app/desktop` directory remains because it contains tracked application source.

Direct removal accepts either a version and optional runtime/target selection,
or the all-releases switch:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 `
  -Action RemoveDesktopRelease -Version 3.3.1 -DesktopRuntime Cpu -DesktopTarget All
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 `
  -Action RemoveDesktopRelease -AllDesktopReleases
```

Portable output is one EXE per variant: the release script streams the audited
ZIP64 runtime onto the PE as an overlay and writes a fixed footer. MSI output
keeps that ZIP as an installer resource. Do not copy only the raw
`app/desktop/build/cargo-target/<variant>/release/xreport-desktop.exe`; it is a shell build,
not a distributable portable artifact.

Release version input is canonicalized in `app/server/pyproject.toml`. The
launcher validates the client package, backend metadata, Cargo package, Tauri
CPU/CUDA configurations, and generated OpenAPI version against that value.

## Manual Backend And Frontend

Non-Windows source-mode commands are development workflows and use the
repository's canonical disposable-cache
root. From the repository root, set `XREPORT_CACHE_ROOT="$PWD/runtimes/cache"`
and the related `XDG_CACHE_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`,
`NPM_CONFIG_CACHE`, `PLAYWRIGHT_BROWSERS_PATH`, `PYTHONPYCACHEPREFIX`, and
`MPLCONFIGDIR` variables below that root before running these commands. The
Windows launcher exports the same locations automatically.

PowerShell:

```powershell
uv run --project app/server python -m uvicorn server.app:app --app-dir app --host 127.0.0.1 --port 5003
Set-Location app/client
npm run dev -- --host 127.0.0.1 --port 8003
```

Use host and port values from `settings/.env`; the Angular development server
proxies `/api` to FastAPI. Run `npm run build` and `npm run preview` only when
validating the built bundle locally or during Tauri development. `npm preview`
is not the production web server. The Windows Launch source-hosting flag is
owned by the launcher and is not part of the manual source workflow.

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
unversioned schema. The launcherâ€™s explicit database option uses the same
coordinator. The `application_settings` singleton is created with typed
defaults on first initialization; after that, settings are read only from the
database. Startup creates required resource directories for logs, models,
tokenizers, checkpoints, and templates. A database re-creation starts with an
empty checkpoint registry; folders already on disk stay unregistered until new
training runs register them again.

## Development Cache Locations

All disposable application, ML, frontend, runtime, and test caches are kept
under the single canonical `runtimes/cache` root. Its subdirectories include
`pytest`, `pytest-tmp`, `ruff`, `python`, `coverage`, `angular`, `uv`, `pip`,
`npm`, `playwright-browsers`, `huggingface`, `torch`, `keras`, and
`matplotlib`. Packaged launches use the same hierarchy below the writable
`<data-root>/runtimes/cache` path; persistent models and application data stay
outside it.

The **Clear cache** action cleans only the canonical `runtimes/cache` directory
(or `<data-root>/runtimes/cache` for packaged launches). It does not search or
remove historical cache locations.

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

## XREPORT Studio launch and application scope

`-Action Launch` (and `-Launch`) starts XREPORT unchanged. `-Action LaunchStudio` starts XREPORT
Studio on `STUDIO_FASTAPI_PORT` (default 5013) with its own launch token, data folder
(`<resources>/studio`), and Angular project (`studio-angular`). The launcher exports `XREPORT_APP`
before starting `server.source_entry` because the data roots are chosen when the server package is
imported. Both applications can run at the same time; each Launch checks only its own port and keeps
the fail-closed handling of unknown listeners.

- Frontend freshness is tracked per project: `dist/<project>/.xreport-build-state.json`, with the
  project name in the Studio fingerprint. Launch builds only the project it needs, and only when stale.
  `-Action RebuildFrontend` builds both unless `-App` is given.
- `-Action KillProcesses` stops every identified XREPORT, Studio, and desktop process; with
  `-App Xreport|Studio` it stops only that application. Studio backends are recognised by
  `server.source_entry --app studio`. Unknown listeners are never stopped.
- `RemoveLogs`, `RemoveCheckpoints`, and `RemoveAllData` act on one application (`-App`, or a prompt in
  the menu). Studio targets live under `<resources>/studio` and never include `models`.
- `-Action MigrateToStudio [-DryRun]` copies existing datasets and checkpoints into Studio (see
  [applications and checkpoint exchange](../architecture/applications_and_checkpoint_exchange.md)).
- Tauri builds (`BuildDesktopRelease`, `LaunchDesktopDev`) remain XREPORT-only.
