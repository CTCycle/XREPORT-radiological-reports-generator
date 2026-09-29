# S51–S53 closure evidence — 2026-09-29

## Revision and environment

- Repository: `G:\Projects\Repositories\Active projects\XREPORT Radiological Reports`
- HEAD: `128e98af4c28bad027cf6bce45c9e7c46baefb87`; working tree was intentionally dirty for this implementation and evidence run.
- Windows 11, AMD Ryzen 5 5600H, NVIDIA RTX 3060 Laptop GPU (6144 MiB), Python `3.14.7`, SQLite disposable root.
- The exact live source runtime used the repository venv and resource root `xreport-live-resources-20260929`.

## S51 — PARTIAL

The maintained harness executed eight source-runtime scenarios: three concurrent validation/evaluation/training batches, processing/evaluation, processing/training, four-way mixed contention, a same-type validation race, and a cancellation request. Validation completed; the same-type race returned the expected `409`; ten independent backend import probes returned `XREPORT Backend`; final running-job inventories were empty; and initial/final SQLite integrity checks were `ok`.

The exact processing lane failed closed after 48.377 seconds because `distilbert-base-uncased` could not be reached and was not cached. The exact checkpoint was not registered and no processed dataset was available, so evaluation, inference, training, and meaningful training cancellation could not be measured. No fallback model or tokenizer was substituted.

The worker lifecycle/failure diagnostics implementation is covered by the focused backend suite: `29 passed`.

Raw evidence: `s51-runtime-receipt-20260929.json`.

## S52 — PASS, browser/source scope

The current source client was built successfully into an isolated writable output and tested through the in-app browser before the automated run. The focused E2E command completed with `2 passed, 14 deselected` in `26.42s`. It covered five routes at five viewports (`320x640`, `390x844`, `640x800`, `1024x720`, `1440x900`) and produced 25 dimension-verified screenshots. It also checked semantic landmarks and names, ARIA references, keyboard-only modal/tour behavior and focus restoration, settings tab arrow navigation, training disclosure keys, reduced motion, responsive reflow, internal catalogue scrolling, visible errors, and browser collectors.

Real Narrator/Speech Recap speech output was not observed and remains a separate non-gating follow-up. The launcher could not overwrite protected canonical Angular `dist` output (`EPERM`); the tested isolated bundle was built from the same current source. This PASS is therefore browser/source scoped and does not cover packaged/native WebView behavior.

Raw evidence: `s52-browser-receipt-20260929.json` and the 25 `s52-*.png` files.

## S53 — PARTIAL

Plan-only generation defines 18 scenarios and generated a deterministic scale-8 manifest with 64 rows. The live run executed ten training/inference baseline, repeat, and long-operation scenarios against the available 8-row fixture. Every training/inference attempt failed closed on missing processed data or the unavailable exact model; no timing or CPU/CUDA performance claim is made. Final running-job inventory was empty and SQLite integrity was `ok`.

The 64-row manifest is preparation evidence only; it was not imported into the application because the exact tokenizer lane was unavailable. Packaged comparison, no-GPU performance, and clinical-quality benchmarking remain outside this run.

Raw evidence: `s53-baseline-receipt-20260929.json` and `s53-scaled-fixture-manifest-20260929.json`.

## Verification commands

- Backend focused tests: `python -m pytest app/tests/unit/test_training_worker_diagnostics.py app/tests/unit/test_training_stop_mechanism.py app/tests/unit/test_job_failure_semantics.py app/tests/unit/test_job_start_concurrency.py app/tests/unit/test_validation_job_semantics.py app/tests/unit/test_job_cancellation_semantics.py -q` → `29 passed`.
- SettingsPage unit test: `ng test --watch=false --include=src/app/pages/settings.page.spec.ts` → `5 passed`.
- Targeted client lint: ESLint on `settings.page.ts` and `settings.page.spec.ts` → pass.
- Isolated production build: Angular application bundle generation → pass.
- S52 browser matrix: recorded above.

## Cleanup

The disposable source-dataset registration was deleted after the run; processed datasets, checkpoints, and running jobs were empty, and the canonical eight fixture images remained intact. The source backend and isolated frontend listeners were stopped after evidence capture.
