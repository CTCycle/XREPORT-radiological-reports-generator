# XREPORT validation — 2026-09-19

## Clean backend smoke

The backend was started from a fresh process on `127.0.0.1:5003` with the
SQLite database migration check enabled. Startup reached Alembic head
`f48a7c2e91b6` and completed resource validation.

The following endpoints returned `200` both sequentially and from a single
concurrent request burst:

- `/api/health`
- `/api/inference/models`
- `/api/preparation/dataset/status`
- `/api/preparation/dataset/names`
- `/api/preparation/dataset/processed/names`
- `/api/training/checkpoints`
- `/api/settings`

Before the final smoke burst, an intentional missing validation-report request
returned the expected `404`; the same endpoint set still returned the statuses
above afterward.

The captured server log is in
[`xreport-backend-smoke-20260919.log`](xreport-backend-smoke-20260919.log).
It contains zero `_ModuleLock`/deadlock, partially initialized torch, circular
Keras/PyTorch, `get_preparation_service`, or HTTP 500 matches. Checkpoint
listing still reports the existing warnings for incomplete `e2e_delete_*`
artifacts, but the endpoint remains successful and no ML import failure is
raised.

## Browser evidence

The rendered Angular E2E suite passed 5/5 against the clean frontend/backend
pair. The responsive evidence test covered model-card grid sizing and banner
dismissal/reset, equal Dataset row geometry, all Settings tabs, control
alignment, narrow layout collapse, light/dark themes, browser console errors,
and failed network requests.

Screenshots:

- [`xreport-inference-narrow-dark.png`](xreport-inference-narrow-dark.png)
- [`xreport-dataset-narrow-dark.png`](xreport-dataset-narrow-dark.png)
- [`xreport-settings-narrow-dark.png`](xreport-settings-narrow-dark.png)
- [`xreport-settings-narrow-light.png`](xreport-settings-narrow-light.png)
