# S40 validation reconciliation — 2026-10-01

Last updated: 2026-10-01

## Decision

S40 is now `PASS` at aggregate gate scope: `10 PASS`, `0 FAIL`, and `0
UNTESTED` across the ten S40 criteria. The native interaction subset is
`PASS` for every exercised scenario (`10/10`), including the controlled
delayed-readiness case.

S40-06 was not independently release-blocking under the current release
policy; it was the final completeness criterion for the aggregate S40 claim.
The overall release remains `NOT APPROVED` for separate reasons: clean final
CPU/CUDA package binding, genuine no-GPU fallback, the full Windows runner,
remaining S51 boundaries, and the approved exact-SHA/hash manifest are still
open. Dirty-tree package diagnostics and the delayed-readiness record exist
separately in the
[packaged-native validation record](packaged-native-validation-20261001.md).

## Reconciled matrix

| Criterion | Current result | Evidence boundary |
| --- | --- | --- |
| S40-01 startup | `PASS` | Real CPU Tauri window reached the ready workspace. |
| S40-02 six native routes | `PASS` | Inference, Reports, Dataset, Training, Settings, and Help were observed. |
| S40-03 refresh/back/forward | `PASS` | Native refresh and history navigation restored the expected route markers. |
| S40-04 settings save/restart | `PASS` | Seed `43` was restored in the second native session. |
| S40-05 keyboard/modal/focus | `PASS` | Keyboard traversal and Help-modal focus containment/restoration passed. |
| S40-06 delayed readiness | `PASS` | The native window showed `XREPORT is still initializing`; the controlled backend started after 18 seconds and the same window reached the ready `Inference` workspace. |
| S40-07 backend stop/error state | `PASS` | Controlled listener stop exposed the native outage state. |
| S40-08 backend retry/recovery | `PASS` | Backend restart, health recovery, refresh, and Settings readiness passed. |
| S40-09 second-instance policy | `PASS` | The second instance showed the policy dialog and exited. |
| S40-10 close/reopen cleanup | `PASS` | Native close exited the shell; the recovery helper was then stopped by exact PID and ports were rechecked clear. |

The earlier [native final receipt](native-recheck-20260930/native-final-validation.json)
records `slow_readiness.status=UNRUN` and `passed=true`; that nine-scenario
receipt remains historical evidence for the first reconciliation and is not an
aggregate S40 claim. The current [S40-06 delayed-readiness receipt](../../desktop/s40-06-delayed-readiness-20261001.json)
records `startup.status=PASS`, `slow_readiness.status=PASS`,
`delayed_backend_seconds=18`, and `ready=true`. The current aggregate combines
the nine previously passed criteria with this independently exercised tenth
criterion.

## Product versus tooling boundary

The client behavior is covered separately by the
[startup-readiness service](../../../../app/client/src/app/services/startup-readiness.service.ts)
and its [unit spec](../../../../app/client/src/app/services/startup-readiness.service.spec.ts):
health polling enters `slow` after 15 seconds, `unavailable` after 60 seconds,
uses the slower unavailable cadence, and recovers to `ready`. The controlled
native run now creates the delayed-backend precondition through the driver's
`StartupBackendDelaySeconds` branch ([driver](../../../../app/desktop/build/validate_native_webview.ps1));
the evidence remains limited to the source CPU development shell and the
specific S40-06 startup/recovery path.

Primary evidence: the [native follow-up](native-follow-up-20260930.md), the
[historical nine-scenario receipt](native-recheck-20260930/native-final-validation.json),
the [current S40-06 receipt](../../desktop/s40-06-delayed-readiness-20261001.json),
the [S40-06 note](s40-06-delayed-readiness-20261001.md), and the
[release-validation record](summary-20260930.md).

## Verification performed for this reconciliation

- The historical native receipt was parsed and counted: its nine requested
  scenarios have `PASS`; its raw `slow_readiness` entry remains `UNRUN` only in
  that superseded receipt.
- The current controlled native receipt was parsed and counted: `startup` and
  `slow_readiness` both have `PASS`, the initial accessible names include
  `XREPORT is still initializing`, the delayed backend interval is `18`
  seconds, and the ready `Inference` route was observed.
- The focused client readiness spec passed: `7` tests.
- The focused desktop packaging/native-driver contract suite passed: `9`
  tests, using an isolated QA temporary root because the host's default global
  pytest root denied directory enumeration.
- PowerShell parsing for `validate_native_webview.ps1` and `smoke_desktop.ps1`,
  plus `git diff --check`, passed.
- No XREPORT process or listener on ports `5003`/`8003`/`8004` remained after
  the validation checks. The controlled native run used an isolated frontend
  port `8004` and did not require administrator elevation.
