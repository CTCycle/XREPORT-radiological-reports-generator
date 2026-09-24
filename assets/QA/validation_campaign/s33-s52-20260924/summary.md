# S33 and S52 Validation Summary

Date recorded: 2026-09-25; browser observations and prerequisite probes: 2026-09-24 UTC.
Source baseline: `994429f6d2f9ad5efee0056486ef6f2bb5795dfc` on `develop`, plus the reviewed working-tree client fixes in this validation bundle.

## Environment and scope

- Windows source-runner workflow with an isolated resources/cache root at `runtimes/cache/s33-s52-20260924`; browser URL `http://127.0.0.1:8003/inference`.
- The model reported `cuda:0` and `cuda_used=true`. The exact GPU model was not recorded. The runtime used the isolated campaign database/resources; canonical model resources were checked read-only.
- Public fixture: `assets/QA/inference_validation_runs/fixture-covid-chest-xray.jpg`, 421,556 bytes, SHA-256 `4570a9524d57cb2697ea577ac5d36553cf28e0d190d347edda1fe12f07d54751`. It is the public `000001-1.jpg` image from `ieee8023/covid-chestxray-dataset`; no direct identifier was observed. Validation was local.
- Exact model: `huggingface:aehrc/cxrmate-multi-tf`, revision `330721b9aa5bba201a3eb88eba4dd9a6607f3e7a`; isolated catalogue reported ready, active, verified integrity, passed technical receipt, and `validation_status=pending`. Output sections were Findings and Impression.

## S33 — PASS for one model and one public image

The in-app browser exercised the rendered flow end to end: image selection, generation and polling, completion, cancellation, retry, report review, provenance, copy, export, edit/save/reload, and persisted Reports history. Successful jobs `699a91f7` and `65cdb860` produced persisted requests `934d348b7529` and `8050f4ec1491`; cancellation jobs `14883ccf` and `ea3ba0ab` reached `cancelled`. The edited report survived reload while the original model output remained available. Provenance showed the exact model revision and CUDA device.

The copied draft contained both output sections (398 characters). Export created `C:\Users\Thomas V\Downloads\xreport-fixture-covid-chest-xray.txt` (398 bytes, SHA-256 `69f6ff150090c7152211cc5d22cf973e8ed560eee61e5c84e8a41723614486d0`); its Findings/Impression contents were checked. The browser automation wrapper did not surface a download event, so the local file itself was verified.

The browser run exposed stale model readiness after terminal generation and generic cancellation messaging. The client now refreshes the model catalogue after terminal jobs and identifies a cancelled generation explicitly. Regression coverage passed: `job-cancellation.pages.spec.ts` (7 tests) and `modal-focus.directive.spec.ts` (1 test). The production client build passed. Logs are [inference client tests](client-unit-inference.log), [modal focus test](client-unit-modal-focus.log), and [production build](client-production-build.log).

This is technical workflow evidence for a single open model and a single public image. It is not a multi-case quality, clinical, or diagnostic-performance result. The model's `validation_status` remains `pending`. The browser screenshot was visually inspected inline; this run did not save a standalone screenshot file.

## S52 — PARTIAL for the focused inference route

Measured viewport checks had no horizontal overflow at 320x720, 768x1024, 1024x720, or 1440x900. Keyboard navigation reached named controls and showed a visible focus indicator. The Help & Tips modal initially allowed Tab to escape while it remained open. `ModalFocusDirective` now creates an attached CDK focus trap; the new unit test verifies both boundary anchors. After a production rebuild, browser retest cycled focus within the 15-control dialog, Escape closed it, and focus returned to the Help and tips trigger.

The stylesheet contains reduced-motion rules, but runtime reduced-motion preference emulation was not performed. The broader Tier 2–3 route, modal, and responsive matrix is also incomplete. The visual state was inspected in the in-app browser but no standalone screenshot was persisted. These limits keep S52 `PARTIAL`.

## Remaining gates rechecked

- **S28 — UNTESTED:** no compatible S24 checkpoint or pinned BEiT encoder was present in the canonical or checked campaign/cache paths. Evaluation was not run.
- **S35 — FAIL / degraded:** the live catalogue remained degraded. The historical three-case canary produced only two distinct reports. Its exact approved fixture directory is absent; no substitute data was used.
- **S37 — BLOCKED:** MedGemma remains gated and not installed. No nonempty `HF_TOKEN` was found in the validation process or `settings/.env`; the stored credential store was not probed. No install or generation was attempted.
- **Canonical CXRMate Multi resource — ISSUE-006 / PARTIAL:** seven of eight files matched the S31 pinned snapshot manifest. `modelling_multi.py` was 18,493 bytes with SHA-256 `4b06e742bbd3f27b6fd61ef0adfcc693ab7b944fc25b41cc5c6cddc45459f5a9`; the manifest expects 17,211 bytes and SHA-256 `528e8d471658802c0ba40fd37a8839cfcd29c342179adeb412f22e04bb7856ee`. The canonical/user resource was not modified. S31's PASS is limited to the fresh isolated install; the canonical offline load is not proof of exact manifest integrity.

## Durable evidence

- [Rendered workflow, device, history, viewport, keyboard, modal, edit, copy, and export observations](ui-observations.json)
- [S28/S35/S37 and live catalogue recheck](gate-recheck.json)
- [Canonical model manifest comparison](canonical-model-recheck.json)
- Client test and build logs linked above

No clinical-quality or release-readiness claim is made. S34 custom-checkpoint inference, S28 evaluation, other-provider browser workflows, inference timeout/persistence-failure cases, S52's remaining accessibility matrix, and broader model-quality gates remain open.
