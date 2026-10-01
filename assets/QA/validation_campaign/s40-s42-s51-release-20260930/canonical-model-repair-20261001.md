# Canonical CXRMate Multi repair — 2026-10-01

Status: `PASS` for canonical resource integrity and one disposable inference
exercise; this does not establish clinical accuracy or release approval.

## Repair and activation

The exact pinned resource was repaired through the supported model-maintenance
API using `model_ref=huggingface:aehrc/cxrmate-multi-tf`, `action=repair`, and
revision `330721b9aa5bba201a3eb88eba4dd9a6607f3e7a`. The maintenance job was
`efd1d7b7`; it downloaded and verified all eight pinned files before the
candidate was activated by the subsequent Generate path (job `91f76020`).

The post-repair comparison against the durable S31 snapshot manifest returned
`all_match=true` for `8/8` files. The previously divergent
`modelling_multi.py` is now `17,211` bytes with SHA-256
`528e8d471658802c0ba40fd37a8839cfcd29c342179adeb412f22e04bb7856ee`. The
remaining pinned hashes are retained in the S31 manifest and the active model
metadata records `integrity=verified`, `state=active`, and the exact active
revision.

## Disposable inference exercise

One supplied QA image (`assets/QA/inference_validation_runs/qa_pa.png`) was
submitted through the supported Generate API with the repaired model and a
deterministic generation profile. The job completed and returned non-empty
Findings and Impression fields. Runtime provenance recorded
`resolved_device=cuda:0`, `cuda_available=true`, and `cuda_used=true` on the
RTX 3060 host. This is a runtime/integrity check only; the generated text is
not a clinical validation claim.

## Boundary reconciliation

`ISSUE-006` is resolved for the current local canonical resource: the installed
model now matches the pinned eight-file manifest and the active metadata points
to the repaired revision. S42 remains `PARTIAL` until the separate genuine
GPU-less packaged lane and the clean-SHA package/release gates are closed.
The disposable data/model state is not itself a release artifact and must not
be inferred as proof of representative-data or model-quality acceptance.
