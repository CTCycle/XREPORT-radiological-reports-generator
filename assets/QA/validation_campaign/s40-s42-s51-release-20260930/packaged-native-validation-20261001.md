# Packaged native validation — 2026-10-01

Status: `PASS` for the exercised packaged CPU/CUDA startup and native
interaction scope; release approval remains open because the artifacts were
built from a dirty working tree and the remaining S42/S51 boundaries are not
closed.

## Candidate and artifacts

The packages were rebuilt from base commit
`e344e40cd129672fbd73a0faba6f568dd83a92f8` with reviewed working-tree repairs
present. Both CPU and CUDA portable/MSI pairs were built and artifact-verified
at version `3.1.0`:

- CPU payload SHA-256: `1682e2fa19bfac675f0923f50a402c7f39be3a7a39b56d6b1e340c260718f710`.
- CUDA payload SHA-256: `03e478659e2c557211f534ebee7d624e737361b9da562ee381508145ee574514`.

Both runtime audits intentionally record `dirty_tree=true`; these are review
artifacts, not the final exact-SHA release bundle.

## Exercised evidence

The current CPU package passed technical startup, readiness, health, frontend
serving, graceful close, backend/listener removal, and contract cleanup in the
[CPU smoke receipt](../../desktop/smoke-cpu-3.1.0.json). The repaired CPU
portable package also has a retained full native receipt with startup, six
routes, refresh/back/forward, keyboard traversal, modal focus, second-instance
policy, and close/port cleanup all `PASS`:
[CPU native receipt](../../desktop/native-package-all-20260930.json).

The current CUDA package passed the same technical contract and the fresh
[CUDA native receipt](../../desktop/native-cuda-retry-20260930.json) records a
real `XREPORT — Radiological Reports (CUDA)` window, six route checks,
refresh/back/forward, keyboard traversal, modal focus, second-instance policy,
and native close/port cleanup all `PASS`. The combined receipt is embedded in
the [CUDA smoke receipt](../../desktop/smoke-cuda-3.1.0.json).

The native driver now refreshes process handles and waits up to its configured
timeout for the real window before attaching. Focus-sensitive checks explicitly
refocus the relevant native control after history navigation. Controlled
delayed readiness is now recorded separately in the [S40-06 receipt](../../desktop/s40-06-delayed-readiness-20261001.json);
the current S40 reconciliation is `10 PASS`, `0 FAIL`, `0 UNTESTED`.

The packaged handoff repair uses `window.location.replace` for the validated
loopback redirect. The packaged session cookie is `SameSite=Lax` for that
top-level WebView2 redirect, and the browser UI's cookie-authenticated health
probe is allowed while shutdown remains private-header-only. Focused security,
packaging, and baseline tests passed (`14` tests in the latest targeted run).

## Remaining release boundaries

- S42 still needs a genuine GPU-less Windows package lane and canonical
  CXRMate Multi `ISSUE-006` reconciliation; the CUDA proof here is RTX 3060
  technical evidence.
- The packaged CPU active-training close/reopen check passed its bounded
  technical scope: it observed `/api/jobs` in `running` state immediately
  before native close, then reopened with health `ok` and zero running jobs.
  See the [active-training note](packaged-active-training-close-reopen-20261001.md)
  and [receipt](packaged-active-training-close-reopen-20261001.json). It used
  an API-started synthetic CPU job, not the native Training form, and does not
  establish graceful user-cancellation semantics.
- S51 remains `PARTIAL` for genuine GPU-less hardware, hosted-CI,
  representative-data, and attribution of the original slow initialization.
- The final clean commit, exact asset hashes, approved release manifest, and
  release-approval workflow have not been performed.

No administrator elevation was required for these packaged checks. All
task-owned XREPORT processes and listeners were absent after validation.
