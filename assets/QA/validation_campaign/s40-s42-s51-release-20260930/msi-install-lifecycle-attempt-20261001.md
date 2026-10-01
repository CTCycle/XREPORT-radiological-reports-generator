# MSI install lifecycle validation — 2026-10-01

Status: `PASS` for CPU and CUDA v3.1.0 MSI install → launch → close → uninstall.

## Historical first two attempts (UNRUN / INCOMPLETE)

Two earlier quiet `/qn` install attempts did not initialize: the task-owned
`msiexec.exe` transaction remained live without creating a log or product
registration and was terminated after its bound. Those attempts established no
MSI behavior and were recorded read-only; they are superseded by the lifecycle
runs below.

## Lifecycle harness

The validation runner `validate_msi_lifecycle.ps1` (helpers under the disposable
`opencode\msi-lifecycle` directory) performs, per variant, on a single elevated
transaction:

1. **install** — `msiexec /i <msi> /qn /norestart ALLUSERS=1 /l*v <log>`;
2. **verify installed** — product registration present, install directory and
   `xreport-desktop.exe` present, `generated\runtime.zip` present;
3. **launch** — start `xreport-desktop.exe` with disposable `LOCALAPPDATA`;
   wait for `desktop-session.json`/`desktop-ready.json`, bootstrap redirect,
   `/api/health` status `ok` with matching variant/version, and a served Angular
   `<app-root` index;
4. **close / cleanup** — graceful close via the backend's own
   `POST /__xreport/shutdown` (the same endpoint the Tauri shell calls) plus
   `CloseMainWindow()`, a 45-second no-force-kill exit poll, and only then a
   forced kill if still alive; verify app exit, backend-process removal,
   listener removal, and contract-file removal;
5. **uninstall** — `msiexec /x <product-code> /qn /norestart /l*v <log>`;
   verify registration and install directory are gone; final port/process
   cleanliness.

Elevation uses `Start-Process -Verb RunAs` launched **detached** and re-triggered
until the UAC prompt is approved; each stage writes a completion marker JSON.

### Harness cleanup fix applied 2026-10-01

The first CPU lifecycle run passed install, registration, launch (health `ok`,
frontend served), uninstall, and all process/listener cleanup but was marked
`FAILED` solely because `contracts_removed=false`: the close path
(`CloseMainWindow` → 20 s → `Stop-Process -Force`) left
`desktop-session.json`/`desktop-ready.json` behind when the shell had to be
force-killed. The harness now requests the app-owned graceful shutdown
(`/__xreport/shutdown`) first, waits up to 45 seconds for a natural exit without
force-kill, and only treats a remaining contract file as a failure after a true
graceful exit. Both reruns below exited gracefully with `forced_kill=false` and
`contracts_removed=true`.

## CPU MSI lifecycle — PASS

- Product: `XREPORT CPU`, version `3.1.0`, product code
  `{EC7D3331-697D-4B7B-8F80-2C27EDE12BB0}`, install dir
  `C:\Program Files\XREPORT CPU`.
- Install `msiexec` exit `0` (`success`); registered; install dir, exe, and
  generated runtime present.
- Launch: health `ok`, runtime variant `cpu`, version `3.1.0`, frontend served,
  backend PID captured, ephemeral port.
- Close: `shutdown_endpoint_ok=true`, `graceful_exit=true`, `forced_kill=false`,
  `contracts_removed=true`.
- Uninstall `msiexec` exit `0`; registration gone, install dir gone.
- Final: `ports_clean=true`, no XREPORT processes.
- Receipt: `assets\QA\validation_campaign\msi-lifecycle-20261001\cpu-receipt-20261001.json`
  (`passed=true`, `stage=passed`). Work logs:
  `runtimes\cache\msi-lifecycle-20261001\msi-cpu-20392\`.

## CUDA MSI lifecycle — PASS

- Product: `XREPORT CUDA`, version `3.1.0`, product code
  `{277D0DE9-DA73-405B-8E5F-31D10E2D76B9}`, install dir
  `C:\Program Files\XREPORT CUDA`.
- Install `msiexec` exit `0` (`success`); registered; install dir, exe, and
  generated runtime present.
- Launch: health `ok`, runtime variant `cuda`, version `3.1.0`, frontend served.
- Close: `shutdown_endpoint_ok=true`, `graceful_exit=true`, `forced_kill=false`,
  `contracts_removed=true`.
- Uninstall `msiexec` exit `0`; registration gone, install dir gone.
- Final: `ports_clean=true`, no XREPORT processes.
- Receipt: `assets\QA\validation_campaign\msi-lifecycle-20261001\cuda-receipt-20261001.json`
  (`passed=true`, `stage=passed`). Work logs:
  `runtimes\cache\msi-lifecycle-20261001\msi-cuda-2880\`.

## Final state

No XREPORT product registration under HKLM/HKCU uninstall keys; no
`C:\Program Files\XREPORT CPU` / `XREPORT CUDA`; no listeners on 5003/8003/8004;
no `xreport` processes.

This establishes MSI installation and uninstallation behavior for both the CPU
and CUDA v3.1.0 packages on this host. It does not establish genuine no-GPU
fallback, hosted-CI coverage, representative performance, or clinical validity.