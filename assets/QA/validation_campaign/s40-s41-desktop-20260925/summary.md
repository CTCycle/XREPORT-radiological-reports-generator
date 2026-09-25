# S40/S41 CPU desktop validation summary

Last updated: 2026-09-25

## Result

- `S40` remains `PARTIAL`: the CPU desktop shell and packaged runtime started
  and closed cleanly, but native WebView route interaction and official
  launcher orchestration were not proven in this run.
- `S41` is `PASS` for the CPU portable and MSI artifact scope: the clean
  official release path built, verified, launched, served the packaged
  Angular/backend surface, and removed its backend, listener, and temporary
  contracts on shutdown.

This is technical software evidence only. It makes no clinical, quality,
representative-data, CUDA, or release-performance claim.

## Revision and environment

- Git source commit: `0ce69d7206051900025bb0fdc42ce2ced544c2ec`
- Branch: `develop`, upstream `origin/develop`
- Release build tree: clean (`dirty_tree=false` in the runtime and release
  metadata)
- OS: Windows 11 Pro 10.0.26200, x64
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU plus AMD Radeon(TM) Graphics
- Runtime: Python 3.14.7, Node.js 22.22.3, Rust toolchain 1.95.0 selected by
  the official release launcher
- Variant: CPU, SQLite packaged data root, MSI WebView2 embed-bootstrapper
  configuration

## Commands and scenarios

1. Focused packaging regression:
   `app/server/.venv/Scripts/python.exe -m pytest app/tests/unit/test_desktop_packaging.py -q --basetemp runtimes/cache/pytest-tmp/s40-s41-desktop-final -o cache_dir=runtimes/cache/pytest/s40-s41-desktop-final`
   Result: `6 passed`.
2. Official release build:
   `./start_on_windows.ps1 -Action BuildDesktopRelease -DesktopRuntime Cpu -DesktopTarget All -Version 3.1.0`
3. Artifact verification:
   `app/desktop/build/verify_desktop_artifacts.ps1 -Variant cpu -Version 3.1.0 -SourceCommit 0ce69d7206051900025bb0fdc42ce2ced544c2ec`
4. Packaged smoke:
   `app/desktop/build/smoke_desktop.ps1 -Variant cpu -Version 3.1.0`
5. Native boundary probe: the portable executable was started with an
   isolated temporary `LOCALAPPDATA`; its expected native window title,
   runtime extraction, database/environment creation, readiness/session
   contracts, backend shutdown, and listener cleanup were observed. See
   [native boundary receipt](native-packaged-boundary.json).

## Durable evidence

- [Focused packaging test log](desktop-packaging-tests-20260925.log)
- [Runtime bundle audit](../../desktop/runtime-cpu-3.1.0.json): 6,764 files,
  1,145,429,733 payload bytes, SHA-256
  `567d1cb9d635d2eac4ceb54da67d5eac504cb719cd28249718ffe3407a417128`.
- [Artifact verification receipt](../../desktop/verification-cpu-3.1.0.json)
- [Packaged smoke receipt](../../desktop/smoke-cpu-3.1.0.json): startup,
  readiness, health, frontend index, graceful close, backend/listener removal,
  and contract removal all `true`; frontend readiness at 27,026 ms and full
  process close at 27,730 ms.
- [Packaged shell log](../../desktop/smoke-cpu-3.1.0-shell.log)
- [Release build metadata](release-metadata-cpu-3.1.0.json)
- [Release checksums](artifact-checksums-cpu-3.1.0.sha256)

The isolated native probe created `.env` and `database.db` below its temporary
data root. The `desktop-session.json` and `desktop-ready.json` contracts were
removed after close, the backend process and listener were absent, and the
temporary root was deleted after the receipt was recorded.

## Remaining limits

- The native Computer Use surface did not expose the Tauri window in this
  session. The authenticated packaged HTTP page could be opened in the
  browser, but its Angular health polling stayed at the startup screen because
  the browser context could not send the private `X-XREPORT-Desktop-Token`
  header; this is not WebView proof.
- `S40` therefore remains `PARTIAL` pending native WebView navigation/recovery
  and an official `LaunchDesktopDev` run without another task owning the
  listeners.
- `S42` remains `UNTESTED` for CUDA portable/MSI build, NVIDIA execution,
  device provenance, fallback, and CPU/CUDA separation.
- `S43` remains `PARTIAL`; actual process-tree termination was not exercised
  here. `S50`, `S51`, `S52`, and `S53` retain their existing partial scopes.
