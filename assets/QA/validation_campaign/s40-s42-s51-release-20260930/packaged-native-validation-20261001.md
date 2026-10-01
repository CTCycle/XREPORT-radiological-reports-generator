# Packaged native validation — 2026-10-01

Status: `PASS` for the exercised packaged CPU startup and the retained CUDA
native-interaction receipt. Current CPU and CUDA release artifact binding is
`PASS`; release approval remains open for the S42, S51, and approval-control
boundaries.

## Candidate and artifacts

The current CPU and CUDA package pairs are bound to commit
`301a5a510e494c501d6c25eb22d5baea824e936a` at version `3.1.0` with
`dirty_tree=false` and have current release-folder artifacts:

- CPU runtime payload SHA-256: `03da18158b95d0e8e7a400f76b4a151f189bce3fd29a2eb69fa8d7fb012631d4`.
- CUDA runtime payload SHA-256: `d69ee87c7f627c277e68e21fcdb49340cb702ac8e2aff402e776a8445ac02f3f`.
- CPU artifact checksums: portable `b15aae37c6efec59d442e54acdd0fbb6480a038036cae6e9d0dce1a6dc8eb1c5`; MSI `a856b320af8e6c02eef323de57fad3ec944a9f41391266c40d53d9b4e50ef2eb`.
- CUDA artifact checksums: portable `c4c6ca6ce7868cb89f0761957e821ee6c97295021be7aea116a08eb1da5c3fe7`; MSI `3cb3aa58c7fa22b29d71b24f45c76aa7533dc6f3e8ee4708cfe6d5991aa4242c`.

Both variants passed the official artifact verifier, including runtime-resource
binding, checksum, portable, and MSI checks. The current release executables
also passed the packaged [CPU/CUDA smoke](current-release-smoke-20261001.md)
for startup, readiness, health, frontend serving, and cleanup. The current
packaged CUDA inference and persisted-history receipt is documented in
[current packaged CUDA inference](packaged-cuda-inference-current-20261001.md).
The retained CUDA native receipt
and inference note remain historical interaction/inference evidence; they are
not relabeled as runs from the current artifact directory.

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

- S42 still needs a genuine GPU-less Windows package lane. The current CUDA
  package now has a clean packaged inference/provenance receipt; the supported repair workflow now resolves
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
- Exact current CPU/CUDA artifact hashes are recorded above. The approved
  release manifest remains intentionally absent until genuine no-GPU evidence
  and current-revision hosted CI are available; the earlier build-environment
  attempts are retained in the [clean-SHA rebinding record](clean-scha-release-rebind-attempt-20261001.md).

An administrator-capable retry was used for build diagnostics, but no MSI
install/uninstall lifecycle was performed. All task-owned XREPORT processes and
listeners were absent after validation.
