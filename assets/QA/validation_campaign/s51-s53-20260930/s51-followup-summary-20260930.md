# S51 follow-up implementation and validation summary — 2026-10-01

Last updated: 2026-10-01

## Outcome

S51 remains `PARTIAL`, but the reported clean-restart behavior is no longer
reproducible in three independent official-launcher source-runtime runs. Each
run completed the maintained eight-scenario S51 matrix, produced real CUDA
training progress and checkpoint artifacts, classified the cancellation case
as `CANCELLED`, left no running jobs, and retained SQLite integrity `ok`.
The current validation follow-up also completed the CPU source lane at both
eight-row and scale-8/64-row fixture sizes, an explicitly emulated
unavailable-GPU fallback lane, and a packaged CPU active-training
close/reopen check. These additions close the runnable source
CPU/scale/fallback comparisons and the bounded packaged cleanup/reopen slice,
but they do not establish genuine GPU-less hardware or a native Training-form
submission/cancellation workflow.

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
- Package source commit: `af666df53b3424c08a279bc9d0b79b81078815bd`.
- The source-lane receipts retain their recorded working-tree provenance. The
  current clean packaged CPU artifact metadata is bound to the package source
  commit with `dirty_tree=false`; the retained CUDA clean-package metadata is
  historical, while the current CUDA diagnostic runtime is dirty-tree-bound.
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

## 2026-10-01 source-lane additions

The three new receipts below were run from disposable resource roots using the
maintained eight-scenario S51 matrix. Each completed all `19` operations with
zero scenario errors, no final running jobs, cleanup errors `0`, and SQLite
`integrity_check=ok`; the cancellation scenario reached its expected terminal
state.

| Receipt | Fixture/lane | Result | Boundary |
| --- | --- | --- | --- |
| [CPU source](s51-cpu-source-run-20261001.json) | 8 rows, `cpu` | `8/8` scenarios completed; six training jobs reached `worker_completed`; one cancellation reached `cancelled`. | CPU requested with `use_device_GPU=false` on the current RTX host; not a no-GPU hardware claim. |
| [CPU scale-8 source](s51-scale8-source-cpu-20261001.json) | 64 rows, `cpu`, clean backend start | `8/8` scenarios completed after the real 64-row upload/load/process path; all scenario running-job checks were empty. | Deterministic generated synthetic scale only; not representative data or packaged training. |
| [Unavailable-GPU emulation](s51-unavailable-gpu-emulated-20261001.json) | 8 rows, `unavailable-gpu` with backend `CUDA_VISIBLE_DEVICES=-1` | `8/8` scenarios completed; device logs recorded `No GPU found. Falling back to CPU` and `CPU is set as the active device`; GPU memory/utilization remained `0`. | Software-masked fallback only. The host sampler still saw the RTX 3060, so this is not genuine GPU-less hardware evidence. |

The scale-8 preparation used the durable
[S53 scale-8 fixture](s53-scale8-fixture/fixture-manifest.json), imported all
`64` rows and matched all `64` images in the disposable root, then verified the
processed baseline at `row_count=64` before S51 execution.

## Packaged active-training close/reopen

The [packaged active-training receipt](../s40-s42-s51-release-20260930/packaged-active-training-close-reopen-20261001.json)
started a CPU training job through the packaged API, observed it in `running`
state, and requested native package close immediately while the job was still
active. The package, backend, listener, and readiness/session contracts were
removed. A second package instance rotated the backend session, returned
health `ok`, and reported zero running jobs before clean close. This is a
technical API-started synthetic close/reopen check; it does not cover the
native Training form, graceful user cancellation semantics, representative
data, or a genuine GPU-less package.

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

The full live-service Python suite was rerun with task-owned basetemp/cache and
QA screenshot roots: `235 passed, 3 skipped, 0 failed, 0 errors` across 238
tests. The packaged CPU active-training close/reopen check is recorded
separately above. The genuine no-GPU lane was not rerun on hardware without a
GPU, and hosted CI for the release revision remains open.

## Remaining limits and status

The following remain outside this follow-up: genuine GPU-less hardware and
packaged/no-GPU training comparison, native Training-form submission and
graceful user cancellation semantics, hosted CI for the release revision,
representative data, and attribution of the original slow
initialization cause. The source-runtime CUDA/CPU/fallback/scale contention
slice and bounded packaged close/reopen slice are now evidenced, but S51 is
kept `PARTIAL` until the remaining broader boundaries receive their own
evidence.
