# Runtime Startup

Last updated: 2026-07-11

## Windows Local Launcher

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1
```

The menu can:

- prepare portable Python, uv, and Node.js in `runtimes/`
- synchronize backend and frontend dependencies
- build and launch the local web application
- initialize the database and run tests
- remove logs, clear caches, or uninstall generated dependencies

The launch option starts the backend, waits for `/api/health`, starts the frontend preview, opens the browser, and then exits the menu.

Set `ALWAYS_REBUILD=true` in `settings/.env` to rebuild the frontend during
application launch. The default `ALWAYS_REBUILD=false` skips that startup
build; the install/update option continues to build the frontend.

## Manual Backend And Frontend

PowerShell:

```powershell
uv run --project app/server python -m uvicorn server.app:app --app-dir app --host 127.0.0.1 --port 5003
Set-Location app/client
npm run preview -- --host 127.0.0.1 --port 8003 --strictPort
```

Use host and port values from `settings/.env`. `VITE_API_BASE_URL` should remain `/api` for the proxied local flow.

## Test Runtime

CMD:

```cmd
app\tests\run_tests.bat
```

The test launcher uses the prepared backend environment and starts required local services when they are not already running.
