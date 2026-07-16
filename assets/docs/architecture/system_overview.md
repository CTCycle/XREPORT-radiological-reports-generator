# XREPORT System Overview

Last updated: 2026-07-16

XREPORT is a local-first client/server system for radiological report generation and model lifecycle workflows.

## Runtime Topology

- Frontend: React + TypeScript + Vite (`app/client`)
- Backend: FastAPI (`app/server`)
- Persistence: SQLite by default, PostgreSQL optional (`app/server/repositories/database`)
- Long-running execution: job manager with start, poll, and cancel contracts (`app/server/services/jobs.py`)
- Local inference: catalog-selected XREPORT checkpoints, Ollama, offline Hugging Face MedGemma, and an opt-in isolated MAIRA-2 worker

## Implementation-Relevant Repository Structure

```text
.
├─ README.md
├─ start_on_windows.ps1
├─ runtimes/
│  ├─ python/
│  ├─ uv/
│  └─ nodejs/
├─ assets/
│  └─ docs/
├─ scripts/
│  └─ windows/
├─ settings/
│  ├─ .env.example
│  └─ configurations.json
└─ app/
   ├─ maira2_worker/
   ├─ resources/
   ├─ scripts/
   │  └─ initialize_database.py
   ├─ server/
   │  ├─ pyproject.toml
   │  ├─ api/
   │  ├─ domain/
   │  ├─ configurations/
   │  ├─ models/
   │  ├─ services/
   │  └─ repositories/
   ├─ client/
   │  ├─ package.json
   │  ├─ vite.config.ts
   │  └─ src/
   └─ tests/
      └─ run_tests.bat
```

## Entry Points

- Backend API entrypoint: `app/server/app.py`
- Frontend web entrypoint: `app/client/src/main.tsx`
- Frontend route composition: `app/client/src/App.tsx`
- Local launcher and maintenance menu on Windows: `start_on_windows.ps1`
