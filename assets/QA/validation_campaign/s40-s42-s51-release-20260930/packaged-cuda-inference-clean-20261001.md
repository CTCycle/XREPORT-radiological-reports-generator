# Clean-SHA packaged CUDA inference — 2026-10-01

Status: `HISTORICAL / NOT CURRENTLY REBINDABLE` for the exercised packaged
CUDA inference scope. The technical run is retained, but it is not current
release-artifact proof.

The retained run reported a clean-source v3.1.0 CUDA portable artifact bound to
`af666df53b3424c08a279bc9d0b79b81078815bd`; its historical release metadata
recorded `dirty_tree=false`, payload SHA-256
`99ec44eb2333d573330abe454d85b181acd363dfdd33653dc01f15417b537448`, and
successful verification for the portable executable and MSI. The corresponding
CUDA files are not present in the current `release/` folder, while the current
runtime audit is dirty-tree-bound with a different payload hash. Rebind this
receipt before using it for release approval.

The existing [packaged inference harness](../s42-cuda-20260927/run-packaged-cuda-inference.ps1)
staged the repaired canonical resource into an isolated data root and used
`qa_pa.png` (`f30ed78a4c18d162dde6e5305116daa1c8cf9cc653eb032495341ba40889c8e8`).
The run recorded:

- exact model revision `330721b9aa5bba201a3eb88eba4dd9a6607f3e7a`;
- exact `modelling_multi.py` SHA-256
  `528e8d471658802c0ba40fd37a8839cfcd29c342179adeb412f22e04bb7856ee`;
- catalog state `ready`, installation `active`, integrity `verified`;
- Generate job `4056199b` completed with one report and runtime provenance
  `resolved_device=cuda:0`, `cuda_available=true`, `cuda_used=true`;
- persisted history status `succeeded` retained the same CUDA provenance;
- package close, backend removal, listener removal, and disposable data-root
  removal all completed.

The legacy receipt file could not be overwritten because its historical QA
path is ACL-protected, so this note preserves the run without relabeling that
older receipt. The harness's `contracts_removed=false` field was recorded
before the disposable root was deleted; the final process/listener/data-root
cleanup checks passed. This is RTX 3060 evidence only and is not a genuine
GPU-less package result, current release-artifact proof, MSI
installation-lifecycle result, or quality claim.
