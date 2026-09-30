# S51 follow-up implementation and validation summary — 2026-09-30

Last updated: 2026-09-30

## Outcome

S51 remains `PARTIAL`, but the reported clean-restart behavior is no longer
reproducible in three independent official-launcher source-runtime runs. Each
run completed the maintained eight-scenario S51 matrix, produced real CUDA
training progress and checkpoint artifacts, classified the cancellation case
as `CANCELLED`, left no running jobs, and retained SQLite integrity `ok`.

The original failure remains preserved in
[s51-source-run4-after-restart.json](s51-source-run4-after-restart.json) and
[s51-backend-restart.err.log](s51-backend-restart.err.log). It did not prove a
model, CUDA, or multiprocessing deadlock: the worker log continued through
dataset, image-path, device, data-loader, model-build, and fit boundaries, but
the API exposed only zero numeric progress and the monitor had no bounded
startup/inactivity failure path. The implementation therefore addresses the
confirmed state/observability/containment defects without changing the model,
checkpoint format, or pinned artifacts. The single underlying runtime reason
for the unusually slow original initialization is still not established.

This is technical evidence against the eight-row generated synthetic fixture.
It is not clinical, diagnostic-performance, representative-scale, packaged
desktop, or release-readiness evidence.

## Revision and environment

- Branch: `develop`.
- Working-tree source revision: `d662c3be83c3d6f6fc5d1c2ea4aeec9a3bfff53c`.
- The receipts record `dirty=true` because the implementation and test edits
  were intentionally kept in the working tree for review.
- Official Windows launcher with an isolated SQLite/resource root and
  `HF_HUB_OFFLINE=1`.
- Windows 11 build `26200`, Python `3.14.7`, 12 logical CPUs, RTX 3060 with
  6144 MiB.
- Canonical resources were not used as the mutable runtime root. The
  disposable runtime was removed after the final run; the JSON receipts retain
  the measured paths and integrity results.

## Exact fixture and training configuration

- Source fixture: `s27_technical_fixture`, eight generated 64x64 rows.
- Processed dataset: `s28_release_20260925`.
- Retained baseline checkpoint: `XREPORT_20260925T141533`.
- One-epoch CUDA training: batch size `1`, one encoder, one decoder,
  embedding dimension `64`, one attention head, `dataloader_workers=0`,
  `jit_compile=false`, mixed precision and augmentation disabled, and the
  pinned BEiT/tokenizer/checkpoint assets.
- Harness polling and resource sampling interval: one second; its 360-second
  scenario deadline remains a validation boundary, not the production
  watchdog setting.

## Implementation result

The working-tree changes provide:

- atomic initial job-result state before the background runner starts;
- worker lifecycle messages correlated by job ID/PID with phase and elapsed
  time, including dataset, device, loader, model, fit, first-batch, epoch,
  checkpoint, and exit boundaries;
- separate lifecycle/numeric-progress and bounded plot IPC handling so plot
  updates cannot drain critical messages;
- monotonic startup, phase-inactivity, first-batch, and cancellation tracking
  with configurable, bounded watchdog deadlines;
- graceful stop, bounded termination, owned process-tree cleanup, typed
  `training_worker_stalled` failures, and cancellation classification that
  remains `CANCELLED`;
- typed failures for nonzero worker exit and missing/empty result payloads;
- first-batch callback instrumentation and API/dashboard phase rendering;
- maintained-harness measurements for phase transitions, first progress, first
  batch where observed, and final worker phase.

No runtime dependency, database migration, model architecture, checkpoint
serialization contract, or pinned training artifact was changed.

## Independent clean-restart receipts

| Receipt | Captured UTC | Successful CUDA training jobs | Expected cancellation | Max first-progress latency | Max first-batch latency observed | Final running jobs | SQLite |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| [run 1](s51-clean-restart-run1-20260930.json) | 2026-09-30 11:50 | 5 | 1 | 17.579s | 17.579s (4/5 sampled) | 0 | `ok` |
| [run 2](s51-clean-restart-run2-20260930.json) | 2026-09-30 11:53 | 5 | 1 | 14.668s | 14.668s (3/5 sampled) | 0 | `ok` |
| [run 3](s51-clean-restart-run3-20260930.json) | 2026-09-30 11:57 | 5 | 1 | 15.814s | 15.814s (5/5 sampled) | 0 | `ok` |

The first-batch sampling gaps are a one-second polling limitation in the
receipt, not evidence that the worker skipped a batch: every successful
training job reported nonzero progress and reached `worker_completed`, while
the worker lifecycle channel recorded the first-batch boundary when it was
observed.

Each receipt also records the eight scenario outcomes, same-type `409` race,
processing/evaluation and processing/training overlap, four-way contention,
cancellation, checkpoint/dataset cleanup, cold-start import probes, resource
samples, and final API/database checks.

## Regression and live checks

- Focused backend S51 regression set: `30 passed`.
- Full backend unit suite in an isolated writable base directory: `180 passed`.
- Ruff on the affected Python implementation, tests, and harness: passed.
- Targeted Pyright on the changed backend modules: `0 errors, 0 warnings,
  0 informations`.
- Angular lint: passed.
- Angular unit suite: `15` files, `49` tests passed, including zero-progress
  initialization, active-phase, terminal-error, and API-to-dashboard lifecycle
  regressions.
- Training API E2E: `4 passed` against the official isolated launcher.
- Official launcher build/readiness: backend health and frontend serving
  reached `200`; the production Angular bundle was current and the launcher
  completed its build/readiness checks.
- In-app Browser inspection rendered the Training page with the independent
  `Current phase`/`Waiting to start` region and accessible worker-status
  semantics while idle.

The standard `app/tests/run_tests.bat` runner and packaged/no-GPU lanes were
not rerun in this follow-up; the prior runner evidence remains historical and
is not promoted to evidence for these uncommitted changes.

## Remaining limits and status

The following remain outside this follow-up: CPU-only and packaged/no-GPU
training comparison, expanded 64-row S51 contention, native WebView
close/reopen behavior during an active training job, and hosted CI for the
working-tree revision. The source-runtime CUDA/SQLite clean-restart and
contention slice is improved and repeatable, but S51 is kept `PARTIAL` until
those broader boundaries and the original slow-initialization cause receive
their own evidence.
