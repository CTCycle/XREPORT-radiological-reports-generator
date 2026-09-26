# S50–S52 current validation — 2026-09-26

Last updated: 2026-09-26

## Result

| Slice | Result | Current evidence | Remaining limitation |
| --- | --- | --- | --- |
| S50 | PARTIAL | The current source launcher reached the built UI and backend on a writable isolated resources root. A settings update to seed 20260926 survived a full listener stop and relaunch; the restarted API returned health 200, the rendered Settings route displayed 20260926, and SQLite integrity was ok. | The protected runtimes/cache root failed before evidence collection, so the aligned run used a repository-local fallback root. No persisted dataset/report was created in this disposable run, and corrupt-database restore/recovery plus user-visible API error presentation remain untested. |
| S51 | PARTIAL | The current checkout passed 25 focused tests, including the atomic concurrent training-start regression, cancellation/failure semantics, validation-job semantics, and image-scanning neighbors. | Only the training same-type race was directly synchronized. Processing, validation, checkpoint-evaluation, UI close/reopen, process-level resource contention, and no-stuck-job scenarios remain open. |
| S52 | PARTIAL | The in-app browser rendered the current Dataset, Training, Reports, Settings, and Inference routes. The Training Tips & Tricks overlay was visually inspected; 16 Tab transitions stayed within its controls and Escape restored focus to Help and tips. The client unit suite passed 43 tests in 13 files. | The browser adapter did not export a standalone screenshot file. Wider viewport/modal coverage and runtime reduced-motion emulation remain untested. |

## Revision and environment

- Source revision: 69737f8f0acec5d225f884cea70e7bb26b2d8664 on develop.
- Host: Windows 11 validation checkout; existing app/server/.venv and current built Angular bundle.
- The requested runtimes/cache disposable root was denied by the existing protected ACL boundary. No protected cache path was changed. The successful run used runtimes/validation-s50-s53-20260926 and removed it after receipt capture.
- The first source launch's automatic browser-open step was denied by the host; the Codex in-app browser opened the UI directly.
- The XREPORT listeners started for this slice were stopped by exact PID and netstat confirmed no remaining listeners on 5003 or 8003.

## Evidence

- [S50 restart receipt](s50-current-restart-receipt.json)
- [S51 regression receipt](s51-current-regression-20260926.md)
- [S52 route/focus observations](s52-route-focus-observations.json)

No production source change was required by these slices. The partial statuses are intentional and preserve the untested boundaries above.
