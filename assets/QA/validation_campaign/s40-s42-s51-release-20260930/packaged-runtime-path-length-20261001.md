# Packaged runtime path-length remediation — 2026-10-01

Status: `PASS` for the reproduced Windows packaged-runtime failure and its
targeted repair. This is diagnostic evidence from a dirty working tree, not a
release approval.

## Reproduction

The CPU portable package was launched with a deliberately deep
`LOCALAPPDATA` root. Before the repair, the extracted runtime directory used
the complete payload SHA-256 as its directory name. The resulting path to
`backend/_internal/regex/_regex.cp314-win_amd64.pyd` was `266` characters and
the packaged backend failed during the `pyi_rth_nltk` hook with the exact
`DLL load failed while importing _regex` error shown in the attached failure
report.

## Repair and proof

`app/desktop/src-tauri/src/runtime.rs` now uses the first 16 hexadecimal
characters of the payload digest as the extracted-runtime cache directory key.
The complete SHA-256 remains in `runtime-manifest.json` and is still checked
before a cache hit is accepted; the shortened key is only a Windows path-length
measure. A Rust unit test covers that the key is short while remaining derived
from the full digest.

The rebuilt dirty CPU diagnostic package reduced the same `_regex` path to
`218` characters. The backend created its session/readiness contracts,
returned health successfully, served the frontend, and closed with no XREPORT
process or listener remaining. The current CPU All rebuild carries payload
SHA-256 `1682e2fa19bfac675f0923f50a402c7f39be3a7a39b56d6b1e340c260718f710`;
the diagnostic deep-root package used during the reproduction carried
`dc6ea8d4886033d7da4a0eda823d08e948097c641d92f6dc31d878a92fba43f9`.

Evidence:

- [CPU runtime audit](../../desktop/runtime-cpu-3.1.0.json)
- [CPU packaged smoke receipt](../../desktop/smoke-cpu-3.1.0.json)
- [CPU artifact verification](../../desktop/verification-cpu-3.1.0.json)
- [runtime cache-key implementation](../../../../app/desktop/src-tauri/src/runtime.rs)

No administrator elevation was required for this reproduction or repair.
