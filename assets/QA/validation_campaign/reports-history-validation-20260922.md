# XREPORT Reports History Validation

Date: 2026-09-22

## Scope

This pass covers the persisted Reports list/detail workflow, section-aware
editing contracts, the desktop inference catalogue geometry fix, and the
reduced startup illustration. It uses the official Windows launcher and the
built frontend at `http://127.0.0.1:8003` with the backend at
`http://127.0.0.1:5003`.

## Automated evidence

- `pytest -c app/server/pyproject.toml app/tests/e2e/test_inference_api.py app/tests/e2e/test_angular_ui.py -q`: **14 passed**.
- `pytest -c app/server/pyproject.toml --basetemp assets/QA/pytest-tmp-full-20260922-final app/tests/unit -q`: **135 passed**.
- `pytest -c app/server/pyproject.toml app/tests/unit/test_repository_persistence.py app/tests/unit/test_database_initialization.py -q`: **17 passed**.
- `pytest -c app/server/pyproject.toml app/tests/unit/test_openapi_schema.py -q`: **3 passed**.
- `npm run test:unit`: **11 test files, 36 tests passed**.
- `npm run lint`: passed.
- `npm run build`: passed; the generated bundle includes lazy `reports-page` and `report-detail-page` chunks.
- Targeted Ruff: passed.
- Server Pyright: **0 errors, 0 warnings, 0 informations**.

The repository-local pytest base directory is intentional: the default
Windows pytest temp root is ACL-protected in this environment and produces
pre-existing access-denied warnings/errors. The final full-unit run passed with
the scoped base directory above.

## Rendered browser evidence

- `/inference` shows the Reports navigation entry and the model catalogue/details
  region remains coherent at desktop width.
- `/reports` rendered the persisted history list with two existing sessions,
  filter controls, status badges, previews, and pagination state.
- `/reports/2967ebf6d889` rendered session metadata, the Findings editor,
  original-output disclosure, and generation metadata/provenance disclosure.
- The disposable seeded browser flow edited a Findings section, reloaded the
  detail route to verify persistence, accepted the explicit delete confirmation,
  and verified the removed card. The fixture was deleted or already absent at
  teardown, leaving the existing user sessions untouched.

## Boundary

This is technical workflow evidence only. It does not establish clinical report
quality, provider-wide Generate/Cancel coverage, packaged CPU/CUDA startup, or
external CI status.
