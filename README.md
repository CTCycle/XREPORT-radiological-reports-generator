# XREPORT Radiological Reports Generator

[![Release](https://img.shields.io/github/v/release/CTCycle/XREPORT-radiological-reports-generator?display_name=tag)](https://github.com/CTCycle/XREPORT-radiological-reports-generator/releases)
[![Python](https://img.shields.io/badge/python-%3E%3D3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Node.js](https://img.shields.io/badge/node.js-22.x-339933?logo=node.js&logoColor=white)](https://nodejs.org/)
[![CTCycle Portfolio](https://img.shields.io/badge/CTCycle-Portfolio-58a6ff?style=flat-square)](https://ctcycle.github.io/CTCycle/)
[![License](https://img.shields.io/github/license/CTCycle/XREPORT-radiological-reports-generator)](LICENSE)
[![CI](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/workflows/ci.yml)

## 1. Project Overview

XREPORT is a client-server research application that generates editable draft radiological reports from X-ray images. Models and generated drafts are not clinically approved and require qualified independent review.
It combines a FastAPI backend and an Angular 22 frontend to support end-to-end workflows for dataset preparation, model training, validation, and report generation.

The application runs locally as a FastAPI backend with an Angular-served web interface. On Windows, `start_on_windows.ps1` manages the portable runtimes, dependencies, and processes.

---

## 2. Model and Dataset (Optional)

XREPORT supports its trained image-captioning checkpoints plus a fixed five-model Hugging Face Transformers catalogue. Selected public models are downloaded on first Generate into the project-local `app/resources` cache, verified, and reused offline on subsequent launches; no separate model server is required.

Supported data sources:
- **MIMIC-CXR** (initial validation dataset)
- **Custom datasets** following the supported image-report pair format

---

## 3. Installation

### 3.1 Windows (One-Click Setup)

Run:
- `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1`

The launcher menu can launch the app, install or update dependencies, initialize the database, run tests, remove logs, clear caches, and uninstall generated dependencies.

### 3.2 macOS / Linux (Manual Setup)

Prerequisites:
- Python 3.14+
- Node.js 22.x + npm
- uv

Setup:
```bash
cd app/server
uv sync
cd app/client
npm ci
npm run build
```

---

## 4. How to Use

### 4.1 Launch

Windows:
- Run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1` and select **Launch application**.

macOS/Linux (manual):
```bash
cd app
uv run --project server python -m uvicorn server.app:app --host 127.0.0.1 --port 5003
cd client
npm run preview -- --host 127.0.0.1 --port 8003
```

### 4.2 Core Workflow

1. Prepare or load dataset on the **Dataset** page.
2. Start training and monitor progress on the **Training** page.
3. Generate draft reports on the **Inference** page.
4. Run dataset/checkpoint validation from **Validation** flows.

### 4.3 UI Snapshots

The snapshots below were captured from the current Windows web interface at 1440×920.

- **Dataset management**: data source selection and dataset processing configuration.
  ![Dataset management page](assets/figures/readme-dataset.png)
- **Training workspace**: training session setup, checkpoint actions, and training dashboard.
  ![Training workspace page](assets/figures/readme-training.png)
- **Inference workspace**: filterable local model catalog, capability-aware study inputs, clinical context, generation profiles, and editable Findings/Impression drafting with copy, regenerate, and export actions.

For operator guidance, see `assets/docs/operations/getting_started.md` and `assets/docs/operations/workflows.md`.

---

## 5. Setup and Maintenance

Use `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1` on Windows to access the consolidated launch and maintenance menu.

---

## 6. Resources

Runtime data is stored under `app/resources`:
- `checkpoints/`
- `logs/`
- `models/`
- `templates/`
- database file (`database.db`)

On Windows, portable runtimes and runtime virtual environment are stored in `runtimes/`.

---

## 7. Configuration

- Runtime/process settings: `settings/.env`
- Backend defaults: `settings/configurations.json`
- Database configuration: `settings/.env`
- Curated local inference catalog: `settings/inference_models.json`

### 7.1 Database initialization behavior

- On backend startup, a missing `settings/.env` is copied from
  `settings/.env.example`. An existing `settings/.env` is never overwritten,
  and `.env` files are excluded by `.gitignore`.
- SQLite mode (`EMBEDDED_DATABASE=true`):
  - On application startup, if `app/resources/database.db` does not exist, the app initializes the SQLite schema automatically.
  - If the file already exists, startup does not recreate, reset, reseed, or cross-validate it.
- PostgreSQL mode (`EMBEDDED_DATABASE=false`):
  - Normal startup only verifies a connection to the configured database and never creates or initializes it.
  - Select option `3` in `start_on_windows.ps1` to create the configured database and schema explicitly.

See also `assets/docs/` for architecture, runtime, operations, and troubleshooting guidance.

---

## 8. Development Status

This project is under active development and may contain incomplete features. Tagged releases (currently v2.4.0) are stable for local evaluation and testing.

## 9. License

This project is licensed under the MIT License. See `LICENSE`.
