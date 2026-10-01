# Settings migration validation

Last updated: 2026-09-17

## Automated checks

- Backend unit suite: `102 passed`.
- Application-settings migration/repository checks: `9 passed`.
- Database initialization and typed catalogue focused checks: `16 passed`.
- Angular unit tests: `18 passed`.
- Angular lint: passed.
- Angular production build: passed.
- Python compileall and Ruff checks: passed (Ruff emitted only existing protected-cache access warnings in the broad scan).
- PowerShell launcher parse: passed.
- `git diff --check`: passed.
- Production-source audit: no retired JSON filenames, loaders, or configuration symbols remain. The only remaining legacy filename references are the bounded Alembic importer and migration/packaging tests.

## Browser workflow

An earlier local browser run against `http://127.0.0.1:8003` with the backend at
`http://127.0.0.1:5003` verified the existing Settings route and persistence flow:

1. Existing footer Settings gear is enabled and routes from `/inference` to `/settings`.
2. Settings page renders General, Data access, and Advanced sections with the four public controls.
3. Seed changed from `42` to `123` and saved successfully.
4. Browser reload returned the persisted value `123`.
5. Reset to defaults returned the seed to `42`.
6. Direct `GET /api/settings` returned only the four public groups and no hidden inference or infrastructure fields.

The browser flow was not rerun after the final source cleanup because no live
application processes were left running. The obsolete JSON files are now deleted;
the migration tests cover import, defaults, invalid legacy values, and rollback.

## Known verification limitation

`cargo check` reaches the Tauri crate but currently fails in the pre-existing
Tauri configuration with `capability with identifier default not found`. The
settings migration changes do not touch that capability configuration.
