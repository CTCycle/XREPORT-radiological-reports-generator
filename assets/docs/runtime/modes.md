# Runtime Modes

Last updated: 2026-07-11

## Supported Modes

### Local Web Mode

- Backend: FastAPI in `app/server/app.py`.
- Frontend: Vite preview or dev server in `app/client`.
- The Windows operator flow uses `start_on_windows.ps1`.
- macOS and Linux use the documented manual backend and frontend commands.

### Containerized Runtime

- Not implemented in the current codebase.

## Limitations And Constraints

- First launch can be slow because dependency synchronization includes heavy ML packages.
- Long-running ML tasks are job-based and poll-driven. No production WebSocket API routes are currently exposed.
- Local filesystem browsing is feature-gated by `features.allow_local_filesystem_access`.
