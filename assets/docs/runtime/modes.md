# Runtime Modes

Last updated: 2026-10-28

## Supported Modes

### Local Web Mode

- Backend: FastAPI in `app/server/app.py`.
- Normal Windows Launch: FastAPI serves the built Angular SPA and `/api` from
  one origin; the launcher owns backend readiness and shutdown.
- Development: Angular CLI `start`/`dev` proxies `/api` to FastAPI. The
  lightweight `preview` server validates a built bundle and supports Tauri
  development; it does not host normal production Launch.
- The Windows operator flow uses `start_on_windows.ps1`. macOS and Linux use
  the documented manual development backend and frontend commands.

### Packaged Windows Desktop Mode

- Tauri 2 is the native shell. The MSI carries a verified ZIP64 runtime as a
  Tauri resource; the single-file portable EXE carries the same archive as a
  streamed PE overlay. Both contain the frozen FastAPI backend and production
  Angular files without asking `rustc` to load the CUDA archive into memory.
- CPU and CUDA are separate products (`io.github.ctcycle.xreport.cpu` and
  `io.github.ctcycle.xreport.cuda`) and use different MSI upgrade codes.
- Both products use `%LOCALAPPDATA%\XREPORT\data` for mutable state and a
  common `Local\io.github.ctcycle.xreport.desktop` mutex.
- The backend selects an available loopback port, writes a short-lived
  readiness contract, and is authenticated by a per-launch token. The shell
  performs readiness/health polling and graceful shutdown.
- The portable executable uses the system WebView2 runtime. The default MSI
  embeds the WebView2 bootstrapper; an offline installer is an explicit build
  option.
- Tauri release serves its bundled `ui/` assets. Rust owns the packaged backend
  sidecar lifecycle; source-mode FastAPI static hosting is not added for Tauri.

### Containerized Runtime

- Not implemented in the current codebase.

## Limitations And Constraints

- First launch can be slow because dependency synchronization includes heavy ML packages.
- Long-running ML tasks are job-based and poll-driven through the generic
  `/api/jobs` resource. No production WebSocket API routes are currently
  exposed.
- Local filesystem access is limited only by operating-system permissions; there is no application-level access setting.
- External inference uses the embedded Hugging Face Transformers provider; no Ollama, llama.cpp, vLLM, or separate model server is required.
- Each Hugging Face entry requires a previously cached snapshot and the exact
  commit declared in the typed catalogue at
  `app/xreport_inference/configurations/inference_models.py`; mutable refs and network
  resolution are rejected.
- All inference models and generated drafts are for research use only and are not clinically approved.
