# Getting Started

Last updated: 2026-08-21

This guidance is for radiology and ML users running local report-generation workflows, plus technical operators validating datasets, training runs, and model outputs.

## Startup Paths

### Windows Local Launcher

1. Run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1`.
2. Select **Launch application**.
3. Wait for runtime, dependency, database migration, build, and health checks to complete.
4. Use the browser opened at the URL configured by `UI_HOST` and `UI_PORT` in `settings/.env`.

For direct launch without the menu, run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action Launch`.

The first launch creates the configured SQLite database (or PostgreSQL database
when permitted) and applies all checked-in Alembic revisions. Subsequent
launches check `alembic_version` and apply only pending upgrades. To run the
same operation without launching the UI, choose **Initialize database** or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action InitializeDatabase
```

### macOS Or Linux Manual Flow

1. Start the backend:

```bash
uv run --project app/server python -m uvicorn server.app:app --app-dir app --host 127.0.0.1 --port 5003
```

2. Start the frontend preview:

```bash
cd app/client
npm run preview -- --host 127.0.0.1 --port 8003
```

## First Report-Generation Run

1. Open **Inference** and read the research-use warning.
2. Select a ready public model or a complete Custom XReport checkpoint. A public
   model that is not cached is downloaded and verified during its first
   **Generate** action; gated models also require the provider's access terms
   and credentials.
3. Add a de-identified radiograph or study, choose a supported generation
   profile, and submit the background job.
4. Review the editable **Findings** and **Impression** fields and inspect the
   returned model, provider, revision, profile, and output metadata before
   copying or exporting the draft.

Generated reports are research-use drafts, not clinically approved reports.
