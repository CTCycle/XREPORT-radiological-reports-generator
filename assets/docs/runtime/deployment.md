# Runtime Deployment

Last updated: 2026-10-21

## Deployment Scope

- XREPORT is a local client/server application.
- Windows users run the application through `start_on_windows.ps1`.
- macOS and Linux source instructions use a manually started development
  backend and Angular development server.

## Windows Runtime Preparation

1. Run `start_on_windows.ps1`.
2. Select **Install / update dependencies** to prepare Python, uv, Node.js, and the frontend build.
3. Select **Launch application** to start FastAPI, which serves the built
   frontend and API from one origin. The launcher stays open as lifecycle owner.

## Runtime Prerequisites

- Windows prerequisites are downloaded into `runtimes/` by the launcher.
- Manual environments require Python 3.14.7, uv 0.11.9, Node.js 22.22.3/npm
  10.9.8, rustup with Rust 1.95.0, and the Windows MSVC Build Tools/SDK for
  desktop packaging.

## Dependency Consistency

- Python dependencies are synchronized with `uv sync --frozen` from
  `app/server/pyproject.toml` and `app/server/uv.lock`.
- Frontend dependencies are always installed with `npm ci` from
  `app/client/package-lock.json`.
- Desktop Tauri dependencies are always installed with `npm ci` from
  `app/desktop/package-lock.json`.

## Database Migration Lifecycle

- Install/update and every backend startup run the same synchronous Alembic
  coordinator before the application reports readiness.
- Development migrations are generated and reviewed from `Base.metadata`; the
  application only applies checked-in revisions.
- Production upgrades use the shared connection/transaction path, SQLite
  exclusive locking, and PostgreSQL advisory locking. Back up persistent data
  before deploying a release containing a schema migration.
- The frozen backend packages `server/migrations` as immutable runtime data;
  the database remains in the user data root.

## Desktop release layout

Tauri release packages continue to use `frontendDist: "ui"` and the existing
Rust-managed backend sidecar. The release frontend is served by Tauri's bundled
assets; `npm preview` is not used as a production web server.

The Windows release command always synchronizes the locked desktop Python and
Node environments, freezes the backend with the pinned desktop PyInstaller
extra, builds the Angular client once, and streams each variant's runtime into
a ZIP64 archive. The archive audit records the commit, architecture,
dirty-tree state, payload hash, file count, sizes, and largest entries, and
rejects secrets, databases, logs, caches, models, symlinks, and duplicate
members. `runtime.zip` and `ui/` are generated inputs and are never committed.

The canonical command is `start_on_windows.ps1 -Action BuildDesktopRelease`.
`app/desktop` also exposes `npm run tauri:build`, which delegates to that
launcher (CPU by default; pass `-DesktopRuntime Cuda` or `All` after `--`).
Raw `tauri build` is an internal shell operation and is not a complete release
path.

After a release build, verify each variant's checksums, embedded runtime,
metadata, portable payload, and MSI payload, then launch the actual portable
executables through the packaged startup smoke test:

```powershell
.\app\desktop\build\verify_desktop_artifacts.ps1 -Variant cpu -Version 3.2.1 -SourceCommit (git rev-parse HEAD)
.\app\desktop\build\verify_desktop_artifacts.ps1 -Variant cuda -Version 3.2.1 -SourceCommit (git rev-parse HEAD)
.\app\desktop\build\smoke_desktop.ps1 -Variant cpu -Version 3.2.1
.\app\desktop\build\smoke_desktop.ps1 -Variant cuda -Version 3.2.1
```

The smoke command uses a short temporary profile by default and writes its
working reports under `assets/QA/desktop/`. Record the scoped result and any
remaining limit in the campaign ledger; do not retain duplicate receipt files
in the tracked tree. If `-DataRoot` is supplied for preserved runtime evidence,
use a short path: a deeply nested repository path can exceed the Windows DLL
loader's effective path limit for native frozen extensions such as
`regex._regex`, causing a test-harness startup failure even though the same
artifact starts correctly under a normal user profile.

The shell extracts atomically to
`%LOCALAPPDATA%\XREPORT\runtime\<variant>\<version>\<payload-sha256>`. The
backend executable and client are immutable there. First launch atomically
seeds `%LOCALAPPDATA%\XREPORT\data\.env` only when absent, then creates or
upgrades `%LOCALAPPDATA%\XREPORT\data\database.db` and its
`application_settings` row. Settings changes are persisted transactionally in
that database; no JSON settings file is staged or recreated. Logs are bounded
and readiness/session files are removed at shutdown.

CPU uses an isolated target overlay for the official `torch==2.10.0+cpu` and
`torchvision==0.25.0+cpu` wheels. CUDA uses the existing locked cu130 stack;
the development environment is not replaced by the CPU build. Models are
downloaded on demand into user data and are never bundled.

Portable files are unsigned, single-executable PE overlays, and depend on the
maintained system WebView2 runtime. MSI packages are per-machine products with
`runtime.zip` as an immutable Tauri resource and embedded WebView2 bootstrapper
mode by default. The tagged `desktop-release.yml` workflow publishes the
verified CPU/CUDA portable, MSI, checksum, and build-metadata files to the
matching GitHub Release. Signing is deliberately absent.
