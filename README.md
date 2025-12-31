# XREPORT Radiological Reports Generator

## 1. Project Overview
XREPORT is a client-server application that generates radiology reports from X-ray images. The FastAPI backend handles dataset ingestion, preprocessing, training and inference, and stores outputs under the resources directory. The React/Vite frontend provides the UI for configuring runs, monitoring progress, and reviewing outputs. The core workflow is: load labeled image-report data, prepare and validate it, train or load a transformer captioning model, run inference on new images, and review reports and diagnostics.

## 2. Model and dataset
XREPORT uses a transformer encoder-decoder approach for image captioning in the radiology domain. A BEiT-based vision encoder extracts image features and the decoder generates report text token by token, using a configurable Hugging Face tokenizer. The project was initially built around the MIMIC-CXR dataset but can be adapted to any dataset of X-ray images paired with report text.

## 2. Installation

### 2.1 Windows (One Click Setup)
The Windows setup is automated via `XREPORT/start_on_windows.bat`. The launcher performs the following actions:

1. Downloads a portable Python 3.13.1 runtime, uv, and Node.js into `XREPORT/resources/runtimes`.
2. Syncs backend dependencies from `pyproject.toml` using uv (optionally with extras).
3. Prunes the uv cache to keep the runtime folder small.
4. Starts the FastAPI backend with uvicorn using settings from `XREPORT/settings/.env`.
5. Installs frontend dependencies, builds the UI, starts the Vite preview server, and opens the browser.

First run behavior: it downloads runtimes and installs all dependencies, so it can take a few minutes. Subsequent runs reuse the local runtimes and only re-sync when dependencies change; the frontend build is reused if `XREPORT/client/dist` already exists.

Portability and side effects: everything is stored inside the project folder (portable runtimes under `XREPORT/resources/runtimes`, Python env under `.venv`, and frontend artifacts under `XREPORT/client`). No system-wide installs are performed.

### 2.2 macOS / Linux (Manual Setup)
Prerequisites:
- Python 3.13.x
- Node.js 22.x and npm
- uv (recommended for dependency management)

Setup steps:
1. Review and edit `XREPORT/settings/.env` to match your environment.
2. Backend setup: from the repository root, run `uv sync` to install Python dependencies.
3. Frontend setup: run `npm install` and `npm run build` in `XREPORT/client`.

Optional extras: use `uv sync --all-extras` if you want the test dependencies.

## 3. How to use

### 3.1 Windows
Double-click `XREPORT/start_on_windows.bat`. The UI opens at `http://127.0.0.1:7861` by default, and the backend listens on `http://127.0.0.1:8000`.

### 3.2 macOS / Linux
Backend:
```bash
uv run python -m uvicorn XREPORT.server.app:app --host 127.0.0.1 --port 8000
```

Frontend:
```bash
cd XREPORT/client
npm run preview -- --host 127.0.0.1 --port 7861 --strictPort
```

URLs:
- UI: `http://127.0.0.1:7861`
- Backend API: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

### 3.3 Using the Application
Use the Dataset area to load and validate labeled image-report pairs, run preprocessing, and confirm the dataset is ready for training and inference.

![Dataset page](XREPORT/assets/figures/dataset_page.png)

Use the Models area to train a transformer model or load a checkpoint, then run inference to generate reports and review evaluation metrics.

![Models page](XREPORT/assets/figures/model_tab.png)

Use the Viewer area to browse images, plots, and report outputs for quick qualitative review.

![Viewer page](XREPORT/assets/figures/viewer_tab.png)

## 4. Setup and Maintenance
`XREPORT/setup_and_maintenance.bat` provides a small maintenance menu for Windows:

- Remove logs: deletes files from `XREPORT/resources/logs`.
- Uninstall app: removes portable runtimes, uv caches, `.venv`, and frontend build artifacts.
- Initialize database: runs the backend database initialization script.

## 5. Resources
`XREPORT/resources` stores runtime assets and generated artifacts so the project remains portable.

- checkpoints: saved model checkpoints and training artifacts created during training and evaluation.
- database: local datasets and the embedded SQLite database (`sqlite.db`) used by the backend.
- logs: application logs for troubleshooting.
- models: cached model artifacts such as encoders and tokenizers.
- runtimes: portable Python, uv, and Node.js installations used by the Windows launcher.
- templates: reserved for template files (the folder exists but is empty in this repo).

## 6. Configuration
Backend configuration lives in `XREPORT/settings/.env` (runtime variables) and `XREPORT/settings/server_configurations.json` (backend metadata, database defaults, and training defaults). Frontend build and preview settings live in `XREPORT/client/vite.config.ts`, while the Windows launcher reads `XREPORT/settings/.env` to control UI host/port overrides.

| Variable | Description |
|----------|-------------|
| FASTAPI_HOST | Backend bind host; defined in `XREPORT/settings/.env`; default `127.0.0.1`. |
| FASTAPI_PORT | Backend port; defined in `XREPORT/settings/.env`; default `8000`. |
| UI_HOST | UI bind host; defined in `XREPORT/start_on_windows.bat` (override via `XREPORT/settings/.env`); default `127.0.0.1`. |
| UI_PORT | UI port; defined in `XREPORT/start_on_windows.bat` (override via `XREPORT/settings/.env`); default `7861`. |
| RELOAD | Backend autoreload toggle; defined in `XREPORT/start_on_windows.bat` (override via `XREPORT/settings/.env`); default `false`. |
| MPLBACKEND | Matplotlib backend for headless rendering; defined in `XREPORT/settings/.env`; default `Agg`. |
| KERAS_BACKEND | Keras runtime backend; defined in `XREPORT/settings/.env`; default `torch`. |
| TF_CPP_MIN_LOG_LEVEL | TensorFlow log verbosity; defined in `XREPORT/settings/.env`; default `1`. |

## 7. License
This project is licensed under the MIT License. See `LICENSE`.
