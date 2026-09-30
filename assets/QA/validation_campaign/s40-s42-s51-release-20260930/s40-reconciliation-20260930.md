# S40 validation reconciliation — 2026-09-30

Last updated: 2026-09-30

## Decision

S40 is `PARTIAL` at aggregate gate scope: `9 PASS`, `0 FAIL`, and `1
UNTESTED` across the ten S40 criteria. The native interaction subset is
`PASS` for every exercised scenario (`9/9`). The single untested criterion is
S40-06, delayed startup readiness.

S40-06 is not independently release-blocking under the current release
policy. It remains a completeness criterion for declaring the aggregate S40
gate complete, so the aggregate stays `PARTIAL` until it is exercised. The
overall release remains `NOT APPROVED` for separate reasons: final CPU/CUDA
package evidence, genuine no-GPU fallback, the full Windows runner, and the
approved exact-SHA/hash manifest are still open.

## Reconciled matrix

| Criterion | Current result | Evidence boundary |
| --- | --- | --- |
| S40-01 startup | `PASS` | Real CPU Tauri window reached the ready workspace. |
| S40-02 six native routes | `PASS` | Inference, Reports, Dataset, Training, Settings, and Help were observed. |
| S40-03 refresh/back/forward | `PASS` | Native refresh and history navigation restored the expected route markers. |
| S40-04 settings save/restart | `PASS` | Seed `43` was restored in the second native session. |
| S40-05 keyboard/modal/focus | `PASS` | Keyboard traversal and Help-modal focus containment/restoration passed. |
| S40-06 delayed readiness | `UNTESTED` | The driver did not delay or suspend the backend before attaching to the startup screen. |
| S40-07 backend stop/error state | `PASS` | Controlled listener stop exposed the native outage state. |
| S40-08 backend retry/recovery | `PASS` | Backend restart, health recovery, refresh, and Settings readiness passed. |
| S40-09 second-instance policy | `PASS` | The second instance showed the policy dialog and exited. |
| S40-10 close/reopen cleanup | `PASS` | Native close exited the shell; the recovery helper was then stopped by exact PID and ports were rechecked clear. |

The raw [native final receipt](native-recheck-20260930/native-final-validation.json)
records `slow_readiness.status=UNRUN` and `passed=true`. `passed` is the
driver result for its nine requested scenarios; it is not an aggregate S40
claim. For the ledger, the unexercised slow-readiness case is reported as
`UNTESTED`, not as a native interaction failure.

## Product versus tooling boundary

The client behavior is covered separately by the
[startup-readiness service](../../../../app/client/src/app/services/startup-readiness.service.ts)
and its [unit spec](../../../../app/client/src/app/services/startup-readiness.service.spec.ts):
health polling enters `slow` after 15 seconds, `unavailable` after 60 seconds,
uses the slower unavailable cadence, and recovers to `ready`. The native
driver's `SlowReadiness` branch only inspects the initial accessible names and
waits for the ready route; it does not create the delayed-backend precondition
([driver](../../../../app/desktop/build/validate_native_webview.ps1)). Therefore
the missing S40-06 observation is a tooling/setup limitation, not evidence that
the product state is broken.

Primary evidence: the [native follow-up](native-follow-up-20260930.md), the
[final native receipt](native-recheck-20260930/native-final-validation.json),
and the [release-validation record](summary-20260930.md).

## Verification performed for this reconciliation

- The checked-in native receipt was parsed and counted: nine requested
  scenarios have `PASS`; `slow_readiness` is the one raw `UNRUN` entry mapped to
  ledger `UNTESTED`.
- The focused client readiness spec passed: `7` tests.
- The focused desktop packaging/native-driver contract suite passed: `9`
  tests, using an isolated QA temporary root because the host's default global
  pytest root denied directory enumeration.
- PowerShell parsing for `validate_native_webview.ps1` and `smoke_desktop.ps1`,
  plus `git diff --check`, passed.
- No XREPORT process or listener on ports `5003`/`8003` remained after the
  validation checks. The native UI driver itself was not rerun in this turn
  because the available computer-use surface exposed no native window; the
  existing native receipt remains the source of the nine native interaction
  results.
