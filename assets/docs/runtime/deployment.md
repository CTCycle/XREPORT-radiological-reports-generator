# Runtime Deployment

Last updated: 2026-07-11

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
