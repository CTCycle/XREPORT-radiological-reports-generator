# S51/S53 current-head and model-catalogue follow-up

Date: 2026-09-29
Repository HEAD: `06f16a9656b6688ec61fc6e3d81c8ec21b6792dc` (develop)
Runtime: official start_on_windows.ps1 -Action Launch, disposable SQLite/resource root, source backend/frontend on 127.0.0.1:5003 / 127.0.0.1:8003
Fixture: pinned eight-row s27_technical_fixture / s28_release_20260925
The runtime receipt was captured before the catalogue edit; the browser/build checks below exercised the subsequent working-tree changes in inference.page.ts, InferencePage.css, and job-cancellation.pages.spec.ts.

## Final status

- S51 remains `PARTIAL`, strengthened by a fresh current-head receipt. Seven batches completed all 15 submitted jobs: three validation/checkpoint-evaluation/CUDA-training triples, processing plus evaluation, processing plus training, and matched CPU/CUDA one-epoch baselines. Every terminal job status was completed, every post-batch and final running-job check was empty, and cleanup preserved only the retained baseline checkpoint and processed dataset.
- S53 remains `PARTIAL`. The current-head samples reached 5,090/6,144 MiB GPU memory, 100% utilization, and 4,414,771,200 bytes of XREPORT-Python working-set memory. These are synthetic technical measurements, not representative, clinical, packaged, or no-GPU performance evidence.
- The earlier cold-start training exit (fadbc216, exit code 1, no child traceback) did not recur in this current-head run. No production fix or PASS claim is made because the failure remains unexplained and non-reproducible.
- The adjacent UI slice is complete: Model catalogue now has a bounded panel and an internally scrollable, keyboard-focusable region. A 40-checkpoint client regression and rebuilt-browser evidence cover the large-catalogue requirement. S52 remains PARTIAL only for the separately missing spoken Narrator/Speech Recap observation.

## Evidence

- [Current-head contention receipt](runtime-contention-receipt-head-20260929.json): 7 batches, 15 jobs, 0 terminal errors, 0 batches with leftover running jobs, and cleanup state.
- [Browser observation](browser-observation-head-20260929.json): rebuilt Dataset, Training, Reports, and Inference routes; live catalogue geometry; internal scroll behavior; and empty browser error/warning collectors.
- Focused backend suite: 24 passed in 7.47s.
- Client focused model-page suite: 9 passed.
- Full client unit suite: 13 files, 45 tests passed.
- Client lint passed. Production build passed on the verbose rerun and produced app/client/dist/client-angular.

## Remaining limitations

Representative or clinical data, larger-scale stress, packaged/native WebView interaction, no-GPU fallback/performance, spoken accessibility output, gated MedGemma access (ISSUE-002), canonical CXRMate Multi resource repair (ISSUE-006), and protected historical cache ownership (ISSUE-004) remain outside this follow-up. The disposable runtime and task-created records were cleaned; the retained baseline records are intentional fixtures.
