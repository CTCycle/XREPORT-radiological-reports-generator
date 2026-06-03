# XREPORT System Overview

Last updated: 2026-06-03

XREPORT is a local-first client/server system for radiological report generation and model lifecycle workflows.

## Runtime Topology

- Frontend: React + TypeScript + Vite (`XREPORT/client`)
- Backend: FastAPI (`XREPORT/server`)
- Desktop wrapper: Tauri 2 (`XREPORT/client/src-tauri`)
- Persistence: SQLite by default, PostgreSQL optional (`XREPORT/server/repositories/database`)
- Long-running execution: job manager with start, poll, and cancel contracts (`XREPORT/server/services/jobs.py`)

## Implementation-Relevant Repository Structure

```text
.
├─ pyproject.toml
├─ README.md
├─ runtimes/
│  ├─ uv.lock
│  └─ .venv/
├─ assets/
│  └─ docs/
├─ release/
│  └─ tauri/
│     ├─ build_with_tauri.bat
│     └─ scripts/
├─ tests/
│  ├─ run_tests.bat
│  ├─ conftest.py
│  ├─ spaserver.py
│  ├─ unit/
│  └─ e2e/
└─ XREPORT/
   ├─ start_on_windows.bat
   ├─ setup_and_maintenance.bat
   ├─ settings/
   │  ├─ .env
   │  └─ configurations.json
   ├─ resources/
   │  ├─ checkpoints/
   │  ├─ logs/
   │  ├─ templates/
   │  ├─ tokenizers/
   │  └─ database.db
   ├─ scripts/
   │  └─ initialize_database.py
   ├─ server/
   │  ├─ app.py
   │  ├─ api/
   │  ├─ domain/
   │  ├─ services/
   │  ├─ repositories/
   │  ├─ learning/
   │  ├─ configurations/
   │  └─ common/
   └─ client/
      ├─ package.json
      ├─ vite.config.ts
      ├─ src/
      │  ├─ main.tsx
      │  ├─ App.tsx
      │  ├─ pages/
      │  ├─ components/
      │  ├─ services/
      │  ├─ hooks/
      │  └─ types/
      └─ src-tauri/
         ├─ Cargo.toml
         ├─ tauri.conf.json
         └─ src/main.rs
```

## Entry Points

- Backend API entrypoint: `XREPORT/server/app.py`
- Frontend web entrypoint: `XREPORT/client/src/main.tsx`
- Frontend route composition: `XREPORT/client/src/App.tsx`
- Desktop entrypoint: `XREPORT/client/src-tauri/src/main.rs`
- Local launcher on Windows: `XREPORT/start_on_windows.bat`
- Desktop build flow on Windows: `release/tauri/build_with_tauri.bat`
