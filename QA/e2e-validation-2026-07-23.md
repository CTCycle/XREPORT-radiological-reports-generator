# XREPORT full-stack validation

Date: 2026-07-23  
Branch: `inference`  
Runtime: backend `127.0.0.1:5003`, frontend preview `127.0.0.1:8003`

## Executive result

The application was launched and exercised through the browser and HTTP API. Eight product/runtime issues were repaired during the audit, and the repaired paths were retested. The final automated result was:

- Backend unit tests: `54 passed`.
- Backend E2E tests: `26 passed, 1 skipped`.
- Backend integration tests: `1 passed, 1 skipped`.
- Frontend production build: passed twice with Vite 5.4.21.
- Final browser console log: empty.

The frontend `lint` script remains unavailable because `eslint` is neither installed in `app/client/node_modules/.bin` nor declared in `app/client/package.json`. No dependency was added during this scoped QA pass. Two generated pytest scratch directories could not be removed because the workspace ACL denied removal even after an elevated attempt; they were left untouched.

## Repaired findings

1. The generated SQLite database was from an older schema and blocked startup. The exact file was backed up to `database-before-schema-repair-20260723-161237.db`, then the ignored generated database was recreated with the repository initializer.
2. `/api/health` did not exist, so the frontend SPA returned HTML with HTTP 200 for that readiness URL. A JSON backend health endpoint was added and verified with HTTP 200.
3. The Vite proxy read settings from `app/settings` instead of the repository-level `settings` directory, causing frontend API calls to return a 500 model-catalog error. The proxy now targets the configured backend.
4. Checkpoint discovery listed incomplete folders containing only a `.keras` file, while metadata loading required the full serialized checkpoint layout. Discovery now requires all four serialized files; the incomplete fixture no longer appears.
5. The inference catalog reported the XREPORT provider as ready when no complete checkpoint existed. It now reports `not_installed` with an explanatory message until a complete checkpoint is available.
6. Dataset form controls had visible labels without programmatic label associations. Stable `id`/`htmlFor` pairs were added; the final audit found no unlabeled controls.
7. The mobile primary navigation hid Dataset and Training behind horizontal scrolling. The navigation now wraps at the mobile breakpoint without document or navigation overflow.
8. `clinical_context` was required by the multipart API even when the selected model did not support it. It is now optional with an empty-string default; the unknown-model E2E scenario and full E2E suite pass.

## Browser and API coverage

- Inference: loaded the workspace, verified catalog/filter interaction, verified disabled generation state, and confirmed the final model catalog response through the frontend proxy.
- Dataset: navigated to the page, uploaded a two-row CSV, verified the browser row/column summary and backend upload log, attempted Load Dataset without an image folder, and verified the explicit validation message plus unchanged empty persisted dataset state.
- Training: navigated to the page and verified that no incomplete checkpoint is exposed; metadata/configuration actions remain disabled until a valid checkpoint exists.
- Invalid requests: verified structured 422 responses for missing inference fields and missing upload files, 404 for an unknown model/checkpoint, and the no-data dataset validation path.
- Persistence: checked dataset status/names after the rejected Load Dataset path and verified that no dataset was persisted.
- Responsive behavior: checked the desktop layout at 1280x720 and the mobile layout at 375x800; both had no horizontal document overflow. The mobile navigation wrapped and exposed all primary links.
- Accessibility: checked visible controls, programmatic labels, and focus visibility. The final Dataset audit found five labeled controls and zero unlabeled controls.
- Runtime evidence: direct API calls, backend access logs, frontend proxy behavior, browser DOM snapshots, screenshots, and browser console logs were used. The final browser console was empty.

## Remaining boundaries

No complete XREPORT checkpoint, usable study-image folder, available Ollama runtime, compatible cached Hugging Face model, or trained dataset was present in the workspace. Therefore successful model inference, training, resume, evaluation, and populated image-folder loading could not be honestly claimed as live E2E results. The readiness, validation, empty-state, invalid-request, and persistence boundaries above were verified instead.

The production build reports only the existing large-chunk warning (`index` approximately 709 kB); it did not fail. The backend log contains expected error-level entries from intentionally invalid upload fixtures in the automated tests.
