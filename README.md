# XREPORT Radiological Reports Generator

[![Release](https://img.shields.io/github/v/release/CTCycle/XREPORT-radiological-reports-generator?display_name=tag)](https://github.com/CTCycle/XREPORT-radiological-reports-generator/releases)
[![Python](https://img.shields.io/badge/python-%3E%3D3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Node.js](https://img.shields.io/badge/node.js-22.x-339933?logo=node.js&logoColor=white)](https://nodejs.org/)
[![CTCycle Portfolio](https://img.shields.io/badge/CTCycle-Portfolio-58a6ff?style=flat-square)](https://ctcycle.github.io/CTCycle/)
[![License](https://img.shields.io/github/license/CTCycle/XREPORT-radiological-reports-generator)](LICENSE)
[![CI](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/workflows/ci.yml)

Last updated: 2026-08-21

## 1. Project Overview

XREPORT is a local-first research application that generates editable draft radiological reports from X-ray images. Models and generated drafts are not clinically approved and require qualified independent review.
It combines a FastAPI backend and an Angular 22 frontend to take users from source data through dataset preparation, model training, validation, and report generation.

The application runs locally as a FastAPI backend with an Angular-served web interface. On Windows, `start_on_windows.ps1` manages the portable runtimes, dependencies, and processes.

Key capabilities:

- import image folders plus CSV/XLSX report metadata, review matches, and inspect paired records
- clean, tokenize, split, and build training-ready datasets
- configure CPU/GPU training, monitor live metrics and logs, save checkpoints, resume, and evaluate
- use five pinned public Hugging Face report-generation models or locally trained Custom XReport checkpoints
- generate editable Findings and Impression drafts from up to 16 study images, then copy or export them for qualified review
- keep validation, model readiness, provenance, and research-use warnings visible in the workflow


## 2. Model and Dataset (Optional)

XREPORT supports its trained image-captioning checkpoints plus a fixed five-model Hugging Face Transformers catalogue. Selected public models are downloaded on first Generate into the project-local resource cache (under `app/resources` by default), verified, and reused offline on subsequent launches; no separate model server is required.

Supported data sources:
- **MIMIC-CXR** (initial validation dataset)
- **Custom datasets** following the supported image-report pair format



## 3. Installation

### 3.1 Windows (One-Click Setup)

Run:
- `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1`

The launcher menu can launch the app, install or update dependencies, rebuild the frontend only, initialize the database, run tests, remove logs, clear caches, and uninstall generated dependencies.

### 3.2 Windows desktop packages

The repository also contains a Tauri 2 desktop shell under `app/desktop`. It
starts the frozen FastAPI service itself, serves the already-built Angular
client, waits for a dynamic loopback health contract, and shuts the service
down gracefully. The ordinary browser launcher remains unchanged.

Development shell (source backend and visible consoles):

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action LaunchDesktopDev
```

Release packaging is Windows x64 only. A clean tree is required for a normal
release; `-AllowDirtyTree` is an explicit diagnostic override:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 `
  -Action BuildDesktopRelease -DesktopRuntime All -DesktopTarget All -Version 3.0.0
# Diagnostic build of the same four artifacts:
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 `
  -Action BuildDesktopRelease -DesktopRuntime All -DesktopTarget All -Version 3.0.0 `
  -AllowDirtyTree -Force
```

Use `-DesktopRuntime Cpu|Cuda` and `-DesktopTarget Portable|Msi` to narrow a
build. MSI embeds the WebView2 bootstrapper by default; pass
`-OfflineWebView2` only when an offline WebView2 installer is intentionally
available. `-Action RemoveDesktopRelease` removes generated release, staging,
and Cargo output without touching user data.

The interactive launcher exposes the same workflow under **DESKTOP RELEASE**.
Choose **Create release artifacts** or **Remove release artifacts**, enter the
version, and select CPU/CUDA portable or MSI output—or all four packages. The
interactive removal updates the selected variant's checksum and build metadata
sidecars; the direct `RemoveDesktopRelease` action remains the full cleanup for
all release and desktop build output.

The automated GitHub workflow is `.github/workflows/desktop-release.yml`. It
runs for `vX.Y.Z` tags or a manual version dispatch, builds the CPU and CUDA
matrix on Windows, verifies the expected EXE/MSI/checksum files, and uploads
them as workflow artifacts. It does not publish a GitHub Release record; GitHub
still provides the source archives for a tag.

Artifacts are written to `release/` with standard SHA-256 manifests:

* `XREPORT-v3.0.0-windows-x64-cpu-portable.exe` and `.msi`
* `XREPORT-v3.0.0-windows-x64-cuda-portable.exe` and `.msi`
* one `.sha256` file per variant plus a build metadata JSON file

Portable artifacts are self-contained single EXEs: the frozen runtime ZIP is
streamed into a PE overlay. MSI artifacts carry the same verified ZIP as an
immutable installer resource. The raw Tauri shell under
`app/desktop/src-tauri/target` is not itself a portable release artifact.

The builds are currently unsigned. Portable execution requires the maintained
system WebView2 runtime; MSI installation is per-machine and normally needs
administrator approval. CPU and CUDA products have separate identifiers and
shortcuts but share one per-user instance guard, so only one XREPORT desktop
variant can run at a time.

### 3.3 macOS / Linux (Manual Setup)

Prerequisites:
- Python 3.14+
- Node.js 22.x + npm
- uv

Setup:
```bash
cd app/server
uv sync
cd ../client
npm ci
npm run build
```



## 4. How to Use

### 4.1 Launch

Windows:
- Run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1` and select **Launch application**.

macOS/Linux (manual, in two terminals):

Terminal 1, from the repository root:

```bash
uv run --project app/server python -m uvicorn server.app:app --app-dir app --host 127.0.0.1 --port 5003
```

Terminal 2:

```bash
cd app/client
npm run preview -- --host 127.0.0.1 --port 8003
```

Database migrations are applied automatically during installation and backend
startup. To initialize or upgrade the configured database explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1 -Action InitializeDatabase
```

For migration development, run Alembic from `app/server` and review generated
revisions before committing them:

```powershell
uv run alembic -c alembic.ini current --check-heads
uv run alembic -c alembic.ini revision --autogenerate -m "describe schema change"
uv run alembic -c alembic.ini upgrade head
```

### 4.2 Core Workflow

1. On **Dataset**, upload an image folder and report table, load the source, review matched/unmatched rows, and build a processed dataset.
2. On **Training**, select a processed dataset, review the five-step configuration wizard, and start or resume a run while watching live metrics, charts, and logs.
3. On **Inference**, select a public or custom model, add up to 16 images, choose a generation profile, and submit a background generation job.
4. Review and edit the returned **Findings** and **Impression** fields, inspect model/provider provenance, and copy or export the research-use draft.
5. Use the Dataset and Training validation actions to check datasets and checkpoints before drawing quality conclusions.

### 4.3 UI Snapshots

The screenshots below were captured from the current Windows web interface at a consistent 1280×720 desktop viewport. Each image is a focused panel frame rather than a stitched full-page capture, with the relevant scrollable content fit to the frame so text and controls remain readable.

#### Dataset handling

The Dataset workflow keeps the source and processing controls together, then lets users inspect a populated image/report pair before using it downstream.

![Dataset image viewer](assets/figures/readme-dataset.png)

#### Training dashboard

This live eight-epoch session shows 100% progress, labeled loss and accuracy axes, plotted metric points, final metrics, and the session log in one place.

![Populated training dashboard](assets/figures/readme-training.png)

#### Inference workflow

The public model catalogue makes readiness, installation, validation state, anatomy, output sections, and licence visible before use. The selected CXRMate Multi TF model is an open, lightweight chest-X-ray reporter.

![Public inference model catalogue](assets/figures/readme-inference.png)

The workflow keeps two de-identified study images, generation controls, and the multi-view input state visible while the draft is produced.

![Inference image workflow](assets/figures/readme-inference-workflow.png)

The editable review panel shows a generated raw draft after model inference, with model, provider, revision, profile, and output metadata ready for qualified review.

![Inference draft review](assets/figures/readme-inference-report.png)

#### Help & Tips

The Tips & Tricks panel provides contextual onboarding, completed workflow steps, and guidance for resuming from an existing checkpoint.

![Help and tips](assets/figures/readme-help-and-tips.png)

For operator guidance, see [Getting started](assets/docs/operations/getting_started.md) and [Core workflows](assets/docs/operations/workflows.md).



## 5. Documentation and Maintenance

Use `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1` on Windows to access the consolidated launch and maintenance menu.

Deeper documentation:

- [Runtime startup and launcher actions](assets/docs/runtime/startup.md)
- [Runtime configuration and environment variables](assets/docs/runtime/configuration.md)
- [Local inference models and first-use lifecycle](assets/docs/runtime/local_inference_models.md)
- [Architecture overview](assets/docs/architecture/system_overview.md)
- [Troubleshooting and database initialization](assets/docs/operations/troubleshooting.md)



## 6. Resources

Runtime data is stored under `app/resources` by default. Set
`XREPORT_RESOURCES_DIR` in `settings/.env` to use another absolute path or a
path relative to the repository root:
- `checkpoints/`
- `logs/`
- `models/`
- `templates/`
- database file (`database.db`)

On Windows, portable runtimes and runtime virtual environment are stored in `runtimes/`.

Disposable runtime caches are stored under `runtimes/cache`; pytest, Ruff, and
other development-tool caches are stored under `app/tests/cache`. Use
`.\start_on_windows.ps1 -Action ClearCache` to remove them. Cleanup skips
locked or administrator-protected files and continues with the remaining
artifacts.

Packaged desktop mode never writes mutable state into the installation
directory. Immutable extracted files live under
`%LOCALAPPDATA%\XREPORT\runtime\<cpu|cuda>\3.0.0\<payload-sha256>`. Runtime data,
the seeded `.env`, configuration, SQLite database, logs, checkpoints, model
downloads, tokenizers, templates, and caches live under
`%LOCALAPPDATA%\XREPORT\data`. Data intentionally survives MSI upgrades and
uninstall; remove that directory manually only when a full reset is wanted.


## 7. Configuration

- Runtime/process settings: `settings/.env`
- Backend defaults: `settings/configurations.json`
- Database configuration: `settings/.env`
- Curated local inference catalog: `settings/inference_models.json`
- Optional Hugging Face access token: `HF_TOKEN` for gated models such as MedGemma; do not commit it

### 7.1 Database initialization behavior

- On backend startup, a missing `settings/.env` is copied from
  `settings/.env.example`. An existing `settings/.env` is never overwritten,
  and `.env` files are excluded by `.gitignore`.
- SQLite mode (`EMBEDDED_DATABASE=true`):
  - On application startup, the Alembic coordinator creates a missing database and upgrades an existing database to the checked-in head.
  - An exact known legacy schema may be adopted; ambiguous or modified non-empty databases fail closed without mutation.
- PostgreSQL mode (`EMBEDDED_DATABASE=false`):
  - Normal startup verifies the configured connection and applies pending revisions when permitted.
  - Select **Initialize database** in `start_on_windows.ps1` to run the same explicit initialization path without launching the UI.

Public models may require a first-use download and verification. Keep network access available for a model that is not already cached; gated models additionally require the provider's access terms and credentials. Once verified, the local snapshot is reused on later launches.

See also `assets/docs/` for architecture, runtime, operations, and troubleshooting guidance.

Packaged mode ignores a source-relative `XREPORT_RESOURCES_DIR`. The shell
passes a per-launch 256-bit token to the backend; the one-time bootstrap URL is
stored only in the user-scoped session file and becomes an HttpOnly,
host-only, `SameSite=Strict` cookie. Native health/shutdown probes use a
private header. The packaged server adds same-origin CSP and standard browser
hardening headers, and it exposes no Tauri filesystem, shell, process, or
arbitrary-opener capability to the Angular content.



## 8. Development Status

This project is under active development and may contain incomplete features. The upcoming major release is v3.0.0, which is intended for stable local evaluation and testing.

## 9. License

This project is licensed under the MIT License. See `LICENSE`.
