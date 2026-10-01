# Current release smoke — 2026-10-01

Status: `PASS` for the packaged CPU and CUDA startup/readiness/health/frontend
and cleanup smoke against the current `release/` executables.

The release binaries and runtime audits are bound by their own build metadata
to source commit
`301a5a510e494c501d6c25eb22d5baea824e936a` with `dirty_tree=false`. The smoke
harness was run afterward from repository HEAD
`aa4c5cabccb78321096f9ca2c36866f3a60f0094`; the intervening commit contains
only QA/documentation changes and no application-source changes. The smoke
receipt `source_commit` fields therefore identify the runner revision, while
the artifact source remains the exact build revision recorded above.

| Variant | Session/readiness | Health/frontend | Close/process/listener/contracts cleanup | Timings |
| --- | --- | --- | --- | --- |
| CPU | `PASS` | `PASS` | `PASS` | session `23.595s`; cleanup `24.668s` |
| CUDA | `PASS` | `PASS` | `PASS` | session `34.066s`; cleanup `34.831s` |

The durable receipts are [CPU smoke](../../desktop/smoke-cpu-3.1.0.json),
[CUDA smoke](../../desktop/smoke-cuda-3.1.0.json), [CPU shell log](../../desktop/smoke-cpu-3.1.0-shell.log),
and [CUDA shell log](../../desktop/smoke-cuda-3.1.0-shell.log). No XREPORT
processes or validation listeners remained after either run. This is current
packaged technical evidence; it does not cover genuine GPU-less hardware,
representative data, native Training-form submission/cancellation, or MSI
install/uninstall lifecycle.
