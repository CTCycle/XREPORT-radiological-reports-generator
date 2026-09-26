# S35/S36 validation campaign — 2026-09-26

## Outcome

| Gate | Model/revision | Technical result | Conservative review | Catalogue result |
| --- | --- | --- | --- | --- |
| S35 | CXRMate-ED / `68251c7605067ddbea330413aade032713fd2192` | `3/3` cases completed, input/output contracts passed, but only `2` distinct full reports | Failed on unresolved lateral/normal output collapse | Remains `validation_status=degraded`; `ISSUE-001` remains open |
| S36A | CheXOne / `0c350e6852ea08f9d9baf3b7595c1a10d4849927` | `3/3` Findings-only cases passed; resource and reload/reuse sentinel passed | Passed | `validation_status=passed`, research-only |
| S36B | CXRMate-2 / `aa8e2d16470e20671acf049687b4707c9bf2f2b5` | `3/3` Findings/Impression cases passed; resource and reload/reuse sentinel passed | Passed | `validation_status=passed`, research-only |

The three fixtures were run as independent one-image cases with unique request
and job identifiers, not as a multi-view study. Exact SHA-256 values,
clinical contexts, generation profiles, original/processed dimensions,
provenance, timing, device/dtype, memory, and errors are recorded in the
aggregate receipts.

For S35, the different input bytes, decoded dimensions, processed dimensions,
contexts, profiles, and exact revision provenance rule out the earlier
application routing/preprocessing explanation. The lateral and normal full
Findings/Impression reports still collapse to identical text, so this remains
a model-behavior failure. Prompts were not tuned and no revision was
substituted.

S36A and S36B passed the technical gates and manual review for case
responsiveness, declared section semantics, prompt/reasoning leakage,
pathological repetition, and unresolved collapse. No labelled references were
used, so these results do not establish diagnostic accuracy, clinical safety,
or release readiness.

## Resource and environment boundary

- Source baseline: `develop` at `1ed1696a99fc6420e7ccabdcb87571fa68e787c6`, with the validation tooling and evidence updates in the working tree.
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, driver 610.88; runs resolved to `cuda:0`.
- S35 peak CUDA memory: `830605312` bytes; `torch.float32`; model load count `3`.
- S36A peak CUDA memory: `4687148544` bytes; `torch.bfloat16`; model load count `4` including the reload/reuse repeat.
- S36B peak CUDA memory: `5260947968` bytes; `torch.bfloat16`; model load count `4` including the reload/reuse repeat.
- Isolated resources: `runtimes/validation-s35-s36-20260926/resources`.
- Isolated cache: `runtimes/validation-s35-s36-20260926/cache`; remote-code dynamic modules were redirected there for CXRMate-ED and CXRMate-2.
- Canonical model resources were not modified. MedGemma was not accessed.
- After the receipts were copied, the exact disposable isolated root was removed; ports `5003` and `8003` have no listeners and no validation process remains running.

## Evidence

- [case manifest](case-manifest.json)
- [run metadata](run-metadata.json)
- [installation receipts](installation-receipts.json)
- [S35 technical receipt](s35-cxrmate-ed-technical-receipt.json)
- [S36A technical receipt](s36a-chexone-aggregate-technical-receipt.json)
- [S36B technical receipt](s36b-cxrmate2-aggregate-technical-receipt.json)
- [rendered catalogue observation](browser-catalogue-observation.json)
- [manual review](manual-review.json)
- [exact commands](commands.md)
- Historical one-case receipts are retained as [CheXOne history](historical-chexone-single-case-receipt.json) and [CXRMate-2 history](historical-cxrmate2-single-case-receipt.json).

## Limitations

The first CheXOne attempt encountered a protected canonical cache boundary and
a later transient process failure; the authoritative fresh isolated rerun
completed and passed. The high/very-high resource runs were sequential. The
technical and conservative review scope is not a labelled clinical evaluation.
Ignored model weights, caches, and transient service logs are disposable
runtime state and are not part of the durable committed evidence bundle.

## Final verification

- Focused inference adapter/provider/manifest/catalogue/independent-case/installation unit suite: `56 passed` in a workspace-contained disposable test root. The first repository-cache attempt hit the known protected-cache ACL boundary; the aligned rerun passed without source changes.
- Adjacent S30/request-validation inference API regression: `6 passed` against a disposable isolated resource root. An initial invocation exposed only a pytest/backend resource-root mismatch in the history fixture; the aligned rerun passed without source changes.
- Rendered catalogue review: the current disposable build showed CheXOne and CXRMate-2 with `validation policy=passed`, `validation evidence=passed`, and their declared Findings-only / Findings+Impression output contracts.
- Pyright on the changed validation scripts/tests: `0 errors, 0 warnings, 0 informations`.
- Ruff on the changed scripts/configuration/tests: `All checks passed` using an isolated cache.
- `git diff --check`: passed.
