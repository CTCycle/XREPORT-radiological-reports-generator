# Current clean packaged CUDA inference — 2026-10-01

Status: `PASS` for one technical inference and persisted-history check against
the current clean CUDA portable artifact.

The package was built and verified from source commit
`301a5a510e494c501d6c25eb22d5baea824e936a`; its CUDA runtime payload is
`d69ee87c7f627c277e68e21fcdb49340cb702ac8e2aff402e776a8445ac02f3f` and the
runtime audit reports `dirty_tree=false`.

The current receipt staged the repaired canonical CXRMate Multi resource into a
disposable data root and completed the `qa_pa.png` inference. The catalogue was
`ready`/`active`/`verified`; the job completed with `resolved_device=cuda:0`,
`cuda_available=true`, and `cuda_used=true`; the newest persisted history entry
retained the same CUDA provenance. The observed RTX 3060 Laptop GPU was sampled
before and after the run. Cleanup removed the packaged process, backend,
listener, contracts, and disposable data root.

See the [current inference receipt](packaged-cuda-inference-current-20261001.json).
This is technical synthetic-fixture evidence only. It does not cover genuine
GPU-less hardware, native WebView interaction, representative data, clinical
quality, or MSI install/uninstall lifecycle.
