# XREPORT validation — 2026-09-17

## Scope

Close the two known limitations: the final rendered browser E2E pass and the
Tauri Rust check.

## Evidence

- Live services: backend `http://127.0.0.1:5003/api/health` returned HTTP 200;
  Angular preview `http://127.0.0.1:8003/` returned HTTP 200.
- In-app browser flow: inference rendered; Settings saved seed `123`, kept it
  after reload, and reset it to `42`; Dataset and Training rendered; validation
  of `e2e-missing-dataset` displayed `No data found for dataset:
  e2e-missing-dataset.`.
- Automated browser E2E command:
  `npm run test:e2e` from `app/client` — **4 passed** in 4.99 seconds.
- Tauri command: `cargo check` from `app/desktop/src-tauri` — **passed**.

The browser run emitted one non-functional pytest cache permission warning for
the existing protected `app/tests/cache/pytest` directory; it did not affect
the four passing tests.
