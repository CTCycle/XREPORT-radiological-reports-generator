# Runtime Modes

Last updated: 2026-08-30

## Supported Modes

### Local Web Mode

- Backend: FastAPI in `app/server/app.py`.
- Frontend: Vite preview or dev server in `app/client`.
- The Windows operator flow uses `start_on_windows.ps1`.
- macOS and Linux use the documented manual backend and frontend commands.

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

### Containerized Runtime

- Not implemented in the current codebase.

## Limitations And Constraints

- First launch can be slow because dependency synchronization includes heavy ML packages.
- Long-running ML tasks are job-based and poll-driven through the generic
  `/api/jobs` resource. No production WebSocket API routes are currently
  exposed.
- Local filesystem browsing is feature-gated by `features.allow_local_filesystem_access`.
- External inference uses the embedded Hugging Face Transformers provider; no Ollama, llama.cpp, vLLM, or separate model server is required.
- Each Hugging Face entry requires a previously cached snapshot and an exact commit in `settings/inference_models.json`; mutable refs and network resolution are rejected.
- All inference models and generated drafts are for research use only and are not clinically approved.
