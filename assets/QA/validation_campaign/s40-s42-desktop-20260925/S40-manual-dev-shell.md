# S40 manual Tauri dev-shell evidence

Date: 2026-09-25 (Europe/Rome)

## Result

Recommendation: retain S40 as `PARTIAL` pending native rendered navigation and an exercised data-root boundary. The manual CPU Tauri shell started and closed cleanly. The health endpoint reported `status=ok`, version `3.1.0`, runtime mode `sqlite`, runtime port `5003`. HTTP GETs to `/`, `/inference`, `/settings`, `/reports`, `/dataset`, and `/training` on port 8003 each returned 200 and the Angular app shell (2352 bytes, `text/html; charset=utf-8`). These responses establish route serving, not actual rendered navigation in the WebView.

The process was `xreport-desktop.exe` PID 32876, window title `XREPORT — Radiological Reports (CPU)`, native window handle `5178892`. The isolated `LOCALAPPDATA` was set to this evidence folder's `tauri-localappdata`; that directory was not created. The dev-mode shell skips packaged backend startup, so this run did not exercise persistent app-data writes or packaged data-root isolation. No canonical data was touched.

After Ctrl+C on the task-owned Tauri command, no process command line referenced the isolated Cargo target. Parent-owned listeners remained unchanged: port 5003 PID 7720 and port 8003 PID 38928. No Chrome tabs were opened or changed.

## Exact invocation

From `app/desktop`, with `XREPORT_DESKTOP_DEV=1`, `XREPORT_DESKTOP_VARIANT=cpu`, `CARGO_TARGET_DIR=<this folder>/tauri-dev-target`, `LOCALAPPDATA=<this folder>/tauri-localappdata`, and `RUSTUP_TOOLCHAIN=1.95.0`:

```powershell
$repo=(Resolve-Path -LiteralPath '.').Path
$qa=Join-Path $repo 'assets/QA/validation_campaign/s40-s42-desktop-20260925'
$override=Join-Path $qa 'tauri-dev-no-resources.json'
$log=Join-Path $qa 'tauri-dev-shell-retry.log'
Set-Location -LiteralPath (Join-Path $repo 'app/desktop')
$env:XREPORT_DESKTOP_DEV='1'
$env:XREPORT_DESKTOP_VARIANT='cpu'
$env:CARGO_TARGET_DIR=(Join-Path $qa 'tauri-dev-target')
$env:LOCALAPPDATA=(Join-Path $qa 'tauri-localappdata')
$env:RUSTUP_TOOLCHAIN='1.95.0'
$cmd='node_modules\.bin\tauri.cmd dev --config src-tauri\tauri.cpu.conf.json --config "'+$override+'" --no-watch 2>&1'
& $env:ComSpec /d /s /c $cmd | Tee-Object -FilePath $log
```

The QA-only config sets `bundle.resources` to `[]`. The first attempt without it stopped at Tauri's build-time resource check because `src-tauri/generated/runtime.zip` is absent. No canonical config or generated resource was created. The successful Cargo target and logs are retained beside this note as `tauri-dev-target`, `tauri-dev-shell.log`, and `tauri-dev-shell-retry.log`.

The official launcher action was not run because it would stop listeners on 5003/8003 before starting its own services, and those listeners belonged to the parent task. The manual invocation shares the CPU config and dev shell but is diagnostic evidence, not proof of the official launcher orchestration.

## Remaining S40 evidence

- Exercise visible route navigation and recovery inside the Tauri WebView.
- Exercise startup/shutdown and data-root separation through the official launcher without colliding with another task's services.
- The parent task's browser validation is separate evidence and does not by itself prove WebView navigation or persistence isolation.
