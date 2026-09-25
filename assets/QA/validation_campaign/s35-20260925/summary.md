# S35 — CXRMate-ED sensitivity canary

Date: 2026-09-25
Checkout: `develop`, HEAD `7d8eb47165258bc0d994b3946d75a69940fbf816`; validation fixes are uncommitted.
Runtime: official source launcher, isolated resource root `runtimes/cache/release-validation-20260925/resources`, CUDA RTX 3060 Laptop GPU.

## Result: FAIL; model remains degraded

- Recovered the three historical canary inputs from `assets/QA/inference_validation_runs`. Their SHA-256 hashes match the S35 receipt exactly; the original expected fixture directory is still absent. The hash comparison is recorded in [fixture-and-model-recheck.json](fixture-and-model-recheck.json).
- The legacy canonical CXRMate-ED directory has nine file-hash mismatches despite its old `integrity_metadata=verified` label. It was preserved and not used. The application's visible Download model workflow fetched the exact pinned revision `68251c7605067ddbea330413aade032713fd2192` into the isolated resource root and ran validation against that staged copy.
- The current offline canary completed all 3 real inference cases with the expected image, context, profile, model revision, and CUDA provenance. It returned exit code 1: the lateral and normal fixtures produced identical report text, leaving only 2 distinct reports. The receipt has no runtime errors and keeps `validation_status=degraded` with `sensitivity_canary_failed` on every result. See [current sensitivity receipt](../../inference_validation_runs/cxrmate-ed-sensitivity-20260925T124718Z.json).
- After the run, the rendered inference card showed CXRMate-ED `ready`, `active · verified`, revision `68251c7605067ddbea330413aade032713fd2192`, but `Validation policy: degraded`; Generate remained disabled. The captured browser observation is in [browser-observations.json](browser-observations.json).

The former fixture-path blocker is resolved for this rerun, but the quality gate still fails. Keep the warning and do not promote this model. The next action is a model-side investigation of the duplicate outputs across distinct fixtures; any proposed change must rerun this exact pinned three-case canary and pass all S35 criteria before promotion.

## Limits

This is a three-image technical sensitivity canary, not a representative clinical-quality evaluation. It does not establish diagnostic accuracy or safe clinical use. The old canonical resource mismatch is separately unresolved and was not modified.
