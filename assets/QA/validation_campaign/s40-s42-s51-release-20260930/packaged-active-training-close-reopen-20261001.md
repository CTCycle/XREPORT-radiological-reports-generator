# Packaged active-training close/reopen — 2026-10-01

Status: `PASS` for the bounded packaged CPU technical lifecycle.

The CPU portable package was rebuilt from source commit
`af666df53b3424c08a279bc9d0b79b81078815bd`; its build metadata records
`dirty_tree=false`. The validator launched the package with the staged S27
technical fixture in a disposable data root, uploaded and processed all eight
rows, started `/api/training/start`, and observed the job in `running` state
with worker phase `target_started` before close. The final receipt records
`passed=true` and the worker PID observed before close was `21876`.

The package was closed while the training job was active. The shell exited and
the packaged backend/listener were both gone. A second package instance then
started on a rotated backend session, reported zero running jobs, and did not
resurrect the prior job (`not_found_after_reopen`). The second instance also
closed with shell/backend/listener cleanup passing.

Evidence: [structured receipt](packaged-active-training-close-reopen-20261001.json),
[CPU build metadata](../../../../release/XREPORT-v3.1.0-windows-x64-cpu-build.json).

This closes the technical packaged active-close/reopen slice for an
API-started CPU job and directly exercises the frozen Windows worker spawn
path. It does not claim native Training-form submission, in-app cancellation
semantics, a completed model artifact during shutdown, genuine GPU-less
hardware, hosted CI, representative data, or attribution of the original
slow initialization. The host has an RTX 3060 and the fixture is deterministic
synthetic data. The receipt’s `dirty_tree=true` reflects post-build QA files;
the package itself is bound to the clean source commit above.

No administrator elevation was required and no task-owned XREPORT process or
listener remained after cleanup.
