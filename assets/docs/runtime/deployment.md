# Runtime Deployment

Last updated: 2026-08-20

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
- Manual environments require Python 3.14+, uv, and Node.js 22.x with npm.

## Dependency Consistency

- Python dependencies are synchronized from `app/server/pyproject.toml`.
- Frontend dependencies are installed from `app/client/package-lock.json` when it exists.

## Desktop release layout

The Windows release command freezes the backend with the pinned desktop
PyInstaller extra, builds the Angular client once, and streams each variant's
runtime into a ZIP64 archive. The archive audit records the commit, dirty-tree
state, payload hash, file count, sizes, and largest entries, and rejects
secrets, databases, logs, caches, models, symlinks, and duplicate members.

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
mode by default. Signing is deliberately absent on this host and the CI
workflow fails closed when signing is requested without a valid certificate.

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
