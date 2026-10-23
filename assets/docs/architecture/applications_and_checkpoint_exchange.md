# Applications and Checkpoint Exchange

Last updated: 2026-10-23

XREPORT is two independently launchable local applications in one repository.
This file owns their boundary, data ownership, and the portable checkpoint
contract that connects them.

| | XREPORT | XREPORT Studio |
|---|---|---|
| Purpose | Inference, report history and review, importing models | Dataset import and preparation, validation, training, evaluation, exporting checkpoints |
| Backend profile | `XREPORT_APP=xreport` (default) | `XREPORT_APP=studio` |
| Source port | `FASTAPI_PORT` (5003) | `STUDIO_FASTAPI_PORT` (5013) |
| Angular project / output | `client-angular` -> `dist/client-angular/browser` | `studio-angular` -> `dist/studio-angular/browser` |
| Data root | `XREPORT_RESOURCES_DIR` (default `data/`) | `<resources>/studio/` |
| Packaged (Tauri MSI/portable, CPU/CUDA) | Yes | No, source mode and local web only |

## Backend composition

`server.app.create_app(profile)` assembles only the routers of its profile
(`profile_routers`). Studio routers are resolved by name, so the XREPORT build
never imports them. The profile must be known when the `server` package is
imported (data roots are resolved at import time), so the launcher exports
`XREPORT_APP` before starting `server.source_entry --app <profile>`; the entry
point fails if the two disagree. `desktop_entry` supports only `xreport`.

- XREPORT mounts inference, jobs, settings, database settings, and
  `POST /api/exchange/import`.
- Studio mounts upload, preparation, training, validation, jobs, settings,
  database settings, `POST /api/exchange/export`, and
  `GET /api/exchange/exports/{filename}`.
- Each profile has its own OpenAPI snapshot
  (`app/shared/openapi.xreport.json`, `openapi.studio.json`) and generated
  client types (`types/generated/{xreport,studio}`).
- `server/common/profile_modules.py` lists the Studio-only modules. A unit test
  walks the static import closure of `server.app` and fails if XREPORT can
  reach any of them; the PyInstaller spec excludes the same list.
- Modules under `server/models/training/` that checkpoints need at load time
  (`layers`, `metrics`, `scheduler`, `encoder`, `processing`, `dataloader`)
  stay shared and in place, because Keras serialises their registered names.

## Data ownership

- XREPORT keeps its existing database, reports, installed models, and
  settings untouched; packaged upgrades therefore keep their history.
- Studio uses `<resources>/studio/` for its database, `database-config.json`,
  checkpoints, exports, and logs.
- The pinned BEiT encoder and tokenizers (`<resources>/models`) are shared,
  immutable resources installed atomically. Studio resolves `MODELS_DIR` to the
  base folder; it never writes elsewhere in XREPORT's tree.
- A process lock (`.xreport.lock` in each data folder) makes a second process on
  the same folder fail at startup instead of mutating the same SQLite file.
  The OS releases it when the owner exits or crashes.
- Launcher cleanup is application-scoped (`-App Xreport|Studio`). Studio
  targets never include `models`. XREPORT "remove all data" still removes the
  shared `models` folder (existing behaviour) and says so in its prompt; it
  never touches `studio/`.

### Migrating existing data into Studio

`app/scripts/migrate_to_studio.py` (launcher: `-Action MigrateToStudio`,
`-DryRun`, or the menu entry) copies `datasets`, `dataset_versions`,
`dataset_records`, `processing_runs`, `training_samples`, `validation_runs`,
`checkpoints`, and `checkpoint_evaluations` rows plus the registered checkpoint
folders into `studio/`. It never changes or deletes legacy data, so XREPORT
keeps its checkpoints and report history. Safeguards: SQLite only, dry run with
row counts and disk estimate, a backup of the legacy database in
`<resources>/backups/`, refusal when Studio already contains rows, one
transaction for rows, temporary folder copies renamed into place, a foreign-key
check, and `studio/migration-manifest.json` as the completion marker (reruns
report "already migrated"). Rollback: stop Studio and delete `studio/`.
Dataset image paths are not rewritten because source images are not copied.

## Checkpoint exchange (`.xreportckpt`, format version 1)

A zip with an allow-listed layout:

```
manifest.json
saved_model.keras
configuration/{configuration,metadata,session_history}.json
tokenizer/<files>          bundled tokenizer (saved with save_pretrained)
encoder/<files>            optional; only when exported with the encoder
```

The manifest (`server/domain/checkpoint_exchange.py`) records the format and
version, architecture id `xreport-beit-transformer`, `custom_objects_api`,
checkpoint name, `content_sha256` (the checkpoint identity), per-file size and
SHA-256, producer versions (application, Keras, Torch), tokenizer name, encoder
repository and pinned revision, and minimal provenance (creation time, epochs,
last losses). It contains no dataset records, paths, or credentials.

Encoder policy: the BEiT encoder (`microsoft/beit-base-patch16-224`, pinned
revision) is loaded offline from `<resources>/models/XRAYEncoder`. The manifest
references it; export can bundle it (`include_encoder`) for offline machines.
Import uses the installed copy, or installs the bundled copy after checking
hashes. If neither exists the import stops with an explicit "encoder is not
installed (offline)" error. Tokenizers import into
`models/tokenizers/_exchange/<name>` and `TokenizerHandler` prefers that folder,
so imported models run without network access.

Import (`CheckpointExchangeService`) is a cancellable `checkpoint_import` job:

1. Synchronous assessment on upload: zip structure, allow-listed member names,
   entry/size/ratio limits (128 entries, 6 GiB total, 1000x ratio), no
   traversal, absolute, backslash, drive, duplicate, encrypted or non-regular
   (symlink) members, manifest/payload agreement, format/architecture/custom
   object compatibility, name conflicts. Invalid input returns 400/409 before
   any job starts.
2. Streaming extraction to `checkpoints/.import-<id>/` with the byte count
   enforced against the manifest, then SHA-256 verification of every file.
3. Installation of bundled tokenizer/encoder (identical content is a no-op,
   different content is a conflict).
4. Smoke test: load the checkpoint with the shared custom objects and run one
   forward pass against the tokenizer vocabulary.
5. Atomic rename into `checkpoints/<name>` and registration in the database.

Failure or cancellation removes the staging folder and anything this import
newly installed, and never touches an existing checkpoint. Startup removes
stale `.import-*` staging. The same content hash for the same name is
"already imported"; a different checkpoint under an existing name is a 409 that
suggests a free name, and the dialog lets the user retry with a new name.
Archives without a manifest but with the four checkpoint files (older XREPORT
versions) import as `legacy` when the pinned encoder is installed; a lone
`.keras` file is rejected with an explanation of the required companion files.

Export (Studio) validates the registered artifact, collects the tokenizer
through `TokenizerHandler`, hashes all members, and writes
`<resources>/studio/exports/<name>.xreportckpt`, offered as a download.

## Not covered

Import from arbitrary third-party frameworks, a live link between the two
applications, and CUDA-specific package acceptance are out of scope or
unverified; see the [status ledger](../project_status_ledger.md).
