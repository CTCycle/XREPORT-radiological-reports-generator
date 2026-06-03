# Runtime Modes

Last updated: 2026-06-03

## Supported Modes

### Local Development Web Mode

- Backend: FastAPI in `XREPORT/server/app.py`
- Frontend: Vite preview or dev server in `XREPORT/client`
- Typical Windows operator flow uses `XREPORT/start_on_windows.bat`

### Desktop Runtime Tauri Mode

- Desktop shell: `XREPORT/client/src-tauri/src/main.rs`
- Bundled with configuration from `XREPORT/client/src-tauri/tauri.conf.json`
- Desktop app starts a local backend process and loads the web UI from local HTTP

### Containerized Runtime

- Not implemented in the current codebase

## Limitations And Constraints

- Desktop mode is currently implemented for Windows in the runtime bootstrap.
- First launch can be slow because dependency synchronization includes heavy ML packages.
- Long-running ML tasks are job-based and poll-driven. No production WebSocket API routes are currently exposed.
- Local filesystem browsing is feature-gated by `features.allow_local_filesystem_access`.
