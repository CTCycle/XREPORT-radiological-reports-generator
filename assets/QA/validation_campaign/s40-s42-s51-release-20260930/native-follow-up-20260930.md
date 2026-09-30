# Native WebView and release workflow follow-up — 2026-09-30

## Scope

This follow-up exercised the CPU Tauri development shell from the official
Windows launcher at working-tree base `9900374ef2e9a37521d0885c297a07d08743260`.
The worktree was intentionally dirty with the implementation changes described
below; this is not final packaged-release evidence.

The Codex computer-use surface did not expose the native Tauri window, so the
live checks used the repository's Windows UI Automation/native-input driver.
This is native desktop evidence, not browser-only evidence.

## Native result

The final combined receipt is
[`native-final-validation.json`](native-recheck-20260930/native-final-validation.json).
Its required scenarios all passed:

- startup and all six route surfaces, including the section-led Dataset page;
- refresh, back, and forward history navigation;
- keyboard traversal across five primary navigation controls;
- Help modal focus containment and restoration;
- controlled backend listener stop with a visible outage state;
- quoted-command backend restart, health recovery, and Settings refresh;
- second-instance policy dialog and exit;
- native window close with the Tauri process exiting.

The cleanup assertion did not require listener cleanup because the recovery
helper was intentionally started as a separate console; it recorded port 5003
at the instant of the assertion. After the launcher completed, the helper was
stopped by exact PID and ports 5003/8003 were verified clear.

The dedicated slow-start/readiness scenario was not run because it requires
attaching while a deliberately delayed backend is still showing the native
startup screen. S40 therefore remains `PARTIAL`.

## Implementation changes

- `validate_native_webview.ps1` now supports explicit scenario groups, comma
  separated CLI selection, native keyboard input, visible route markers,
  descendant focus lookup, refresh/history, modal, backend stop/retry,
  second-instance, slow-readiness, and close/cleanup assertions.
- `smoke_desktop.ps1` forwards scenario groups and close/port-cleanup options.
- `desktop-release.yml` now separates immutable package preparation from the
  later approval/publish run. The approval run reuses retained artifacts and
  verifies the manifest against the immutable package source commit, avoiding a
  self-referential approval commit SHA.
- The manifest verifier's mismatch diagnostic now names the immutable package
  source commit explicitly.

Focused unit validation passed `13` tests covering desktop packaging and the
resilience harness. PowerShell parsing for both desktop drivers and YAML parsing
for the release workflow also passed.

## Remaining boundaries

No final CPU/CUDA package rebuild, MSI clean-install/uninstall check, genuine
GPU-less CUDA fallback, or packaged native WebView run was performed here.
The S51 CPU/unavailable-GPU/scale-8/package matrix and native close during an
active worker were also not run. The release remains `NOT APPROVED` until those
receipts, the full Windows runner, and the approved eight-asset manifest are
available.
