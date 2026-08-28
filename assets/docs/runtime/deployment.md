# Runtime Deployment

Last updated: 2026-08-26

## Deployment Scope

- XREPORT is a local client/server application.
- Windows users run the application through `start_on_windows.ps1`.
- macOS and Linux users start the backend and frontend manually.

## Windows Runtime Preparation

1. Run `start_on_windows.ps1`.
2. Select **Install / update dependencies** to prepare Python, uv, Node.js, and the frontend build.
3. Select **Launch application** to start the backend and frontend locally.

## Runtime Prerequisites

- Windows prerequisites are downloaded into `runtimes/` by the launcher.
- Manual environments require Python 3.14.2, uv 0.11.9, Node.js 22.22.3/npm
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

The shell extracts atomically to
`%LOCALAPPDATA%\XREPORT\runtime\<variant>\<version>\<payload-sha256>`. The
backend executable and client are immutable there. First launch atomically
seeds `%LOCALAPPDATA%\XREPORT\data\.env` and
`%LOCALAPPDATA%\XREPORT\data\settings\configurations.json` only when absent;
later edits are preserved. Logs are bounded and readiness/session files are
removed at shutdown.

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

## DILIGENT reference comparison

XREPORT follows the useful DILIGENT pattern of a frozen backend, verified
runtime extraction, data-root separation, and Windows Job Object. It adds the
controls that the reference implementation still lacks: a common CPU/CUDA
single-instance mutex, splash-before-startup plus actionable failure UI,
authenticated loopback traffic and backend CSP headers, an HTTPS-only external
navigation allowlist, graceful shutdown with a crash watchdog, streamed ZIP64
bundling, exact artifact selection, and standard checksum manifests. These
corrections should also be applied to DILIGENT: its fixed readiness/log files
race on a second launch; the splash is created after a blocking wait; localhost
HTTP lacks equivalent authentication/security headers; rejected arbitrary
schemes are handed to the OS opener; shutdown is force-only; the bundler reads
large files into memory; and release publication can select an arbitrary EXE,
uses size-only MSI validation, duplicates frontend build work, and writes
nonstandard checksum labels. The DILIGENT checkout was not modified.
