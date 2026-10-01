# S40-06 delayed startup readiness — 2026-10-01

## Result

`PASS` for the S40-06 criterion. This is a native CPU Tauri/WebView
observation of delayed local-backend readiness; it is not a packaged-release,
CUDA, no-GPU, clinical, or S51 acceptance result.

The controlled run used the built Angular frontend on loopback port `8004` and
a real `XREPORT — Radiological Reports (CPU)` native window. The native driver
captured the startup screen, waited `18` seconds before starting the source
FastAPI backend on port `5003`, and then observed the same window reach the
ready `Inference` workspace.

Observed receipt fields:

- `passed`: `true`
- `scenarios.startup.status`: `PASS`
- `scenarios.slow_readiness.status`: `PASS`
- `scenarios.slow_readiness.startup_states`: `XREPORT is still initializing`
- `scenarios.slow_readiness.delayed_backend_seconds`: `18`
- `scenarios.slow_readiness.ready`: `true`

## Evidence and cleanup

- [native S40-06 receipt](../../desktop/s40-06-delayed-readiness-20261001.json)
- [captured startup frame](../../desktop/native-screenshots/startup.png)
- [delayed-readiness driver](../../../../app/desktop/build/validate_native_webview.ps1)

The isolated frontend, Tauri process, backend process, and temporary Cargo
target were removed after the run. Ports `5003` and `8004` were rechecked clear.
The driver was invoked with an explicitly quoted `--app-dir` value because the
Windows repository path contains spaces; the receipt is the authoritative
result, and its PASS fields were independently parsed after cleanup.
