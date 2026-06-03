# Getting Started

Last updated: 2026-06-03

This guidance is for radiology and ML users running local report-generation workflows, plus technical operators validating datasets, training runs, and model outputs.

## Startup Paths

### Windows Local Launcher

1. Run `XREPORT/start_on_windows.bat`.
2. Wait for runtime and dependency checks to complete.
3. Open the UI URL configured by `UI_HOST` and `UI_PORT` in `XREPORT/settings/.env`.

### Windows Packaged Desktop

1. Install and launch the packaged desktop app.
2. Wait for first-run backend initialization.
3. Use the embedded desktop UI.

### macOS Or Linux Manual Flow

1. Start the backend:

```bash
uv run python -m uvicorn XREPORT.server.app:app --host 127.0.0.1 --port 5003
```

2. Start the frontend preview:

```bash
cd XREPORT/client
npm run preview -- --host 127.0.0.1 --port 8003 --strictPort
```
