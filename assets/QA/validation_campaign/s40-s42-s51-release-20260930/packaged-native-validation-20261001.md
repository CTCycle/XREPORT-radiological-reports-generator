# Packaged native validation — 2026-10-01

Status: `PASS` for the exercised packaged CPU startup and the retained CUDA
native-interaction receipt. Current release artifact binding remains
`PARTIAL`; release approval is open for the remaining CUDA, S42, S51, and
approval-control boundaries.

## Candidate and artifacts

The current CPU package pair is bound to commit
`af666df53b3424c08a279bc9d0b79b81078815bd` at version `3.1.0` with
`dirty_tree=false` and has current release-folder artifacts:

- CPU payload SHA-256: `e06e0be4fa22427a4e0cb6950c97cbe84f6f468b6fb625a0e909c7b47fc65a21`.

The current CUDA build produced only a dirty-tree diagnostic runtime audit
(`dirty_tree=true`, payload SHA-256
`2d6fc6086a5662cf29c8276a6fa8d063ac10edbc2274d851150a1ec05d7dffbe`) and a
target-folder MSI/raw executable during direct Tauri diagnostics. The CUDA
portable/MSI pair, checksum, and build metadata are not present in the current
`release/` folder, and the wrapper/post-processing step returned `-1`. The
earlier clean-source CUDA receipt and inference note are retained as historical
records, but are not current release-artifact proof.

## Exercised evidence

The current CPU package passed technical startup, readiness, health, frontend
serving, graceful close, backend/listener removal, and contract cleanup in the
[CPU smoke receipt](../../desktop/smoke-cpu-3.1.0.json). The repaired CPU
portable package also has a retained full native receipt with startup, six
routes, refresh/back/forward, keyboard traversal, modal focus, second-instance
policy, and close/port cleanup all `PASS`:
[CPU native receipt](../../desktop/native-package-all-20260930.json).

The retained [CUDA native receipt](../../desktop/native-cuda-retry-20260930.json)
records a real `XREPORT — Radiological Reports (CUDA)` window, six route checks,
refresh/back/forward, keyboard traversal, modal focus, second-instance policy,
and native close/port cleanup all `PASS`. Its recorded package provenance is
historical relative to the current artifact directory, so it is not relabeled
as current clean-release evidence. The combined historical receipt remains
embedded in the [CUDA smoke receipt](../../desktop/smoke-cuda-3.1.0.json).

The native driver now refreshes process handles and waits up to its configured
timeout for the real window before attaching. Focus-sensitive checks explicitly
refocus the relevant native control after history navigation. Controlled
delayed readiness is now recorded separately in the [S40-06 receipt](../../desktop/s40-06-delayed-readiness-20261001.json);
the current S40 reconciliation is `10 PASS`, `0 FAIL`, `0 UNTESTED`.

The packaged handoff repair uses `window.location.replace` for the validated
loopback redirect. The packaged session cookie is `SameSite=Lax` for that
top-level WebView2 redirect, and the browser UI's cookie-authenticated health
probe is allowed while shutdown remains private-header-only. Focused security
and packaging unit tests passed (`11` tests in the final repeat; the broader
targeted baseline was also green).

## Remaining release boundaries

- S42 still needs a genuine GPU-less Windows package lane and a current
  clean-SHA CUDA release pair. The supported repair workflow now resolves
  `ISSUE-006` for the current local canonical resource; see the [canonical
  repair note](canonical-model-repair-20261001.md). The CUDA proof is RTX 3060
  technical evidence; the current dirty diagnostic MSI is not promoted to the
  release folder.
- The packaged CPU active-training close/reopen check passed its bounded
  technical scope: it observed `/api/jobs` in `running` state immediately
  before native close, then reopened with health `ok` and zero running jobs.
  See the [active-training note](packaged-active-training-close-reopen-20261001.md)
  and [receipt](packaged-active-training-close-reopen-20261001.json). It used
  an API-started synthetic CPU job, not the native Training form, and does not
  establish graceful user-cancellation semantics.
- S51 remains `PARTIAL` for genuine GPU-less hardware, hosted-CI,
  representative-data, and attribution of the original slow initialization.
- Exact CPU artifact hashes and the historical CUDA hashes are recorded in the
  clean-SHA rebinding note. The approved release manifest remains intentionally
  absent until a current CUDA pair, genuine no-GPU evidence, and current-
  revision hosted CI are available; the build-environment attempts and direct
  Tauri diagnostic result are in the [clean-SHA rebinding record](clean-scha-release-rebind-attempt-20261001.md).

An administrator-capable retry was used for build diagnostics, but no MSI
install/uninstall lifecycle was performed. All task-owned XREPORT processes and
listeners were absent after validation.
