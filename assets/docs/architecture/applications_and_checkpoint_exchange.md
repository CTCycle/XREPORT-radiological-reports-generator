# Applications and Checkpoint Exchange

Last updated: 2026-10-28

XREPORT is two independently launchable local applications in one repository: XREPORT Inference and XREPORT Studio.
This file owns their boundary, data ownership, and the portable checkpoint
contract that connects them.

| | XREPORT Inference | XREPORT Studio |
|---|---|---|
| Purpose | Inference, report history and review, importing models | Dataset import and preparation, validation, training, evaluation, exporting checkpoints |
| Backend package | `app/xreport_inference` | `app/xreport_studio` |
| Backend entry | `python -m xreport_inference.entry`, ASGI `xreport_inference.app:app` | `python -m xreport_studio.entry`, ASGI `xreport_studio.app:app` |
| Source port | `FASTAPI_PORT` (5003) | `STUDIO_FASTAPI_PORT` (5013) |
| Angular project / output | `client-angular` (`app/client/projects/inference`) -> `dist/client-angular/browser` | `studio-angular` (`app/client/projects/studio`) -> `dist/studio-angular/browser` |
| Data root | `XREPORT_RESOURCES_DIR` (default `data/`) | `<resources>/studio/` |
| Packaged (Tauri MSI/portable, CPU/CUDA) | Yes | No, source mode and local web only |

## Package boundaries

Backend (`app/`):

- `xreport_shared`: infrastructure both applications need: job manager, settings,
  database engine/initializer/migration runner, checkpoint exchange core
  (manifest, archive limits, service), the application factory
  (`app_factory.create_application`), source runner, and the model code a
  checkpoint needs at load time (`models/training/{layers,metrics,scheduler,encoder,processing,dataloader}`,
  `models/inference/generator`). Keras serialises the registered layer names, so
  these classes stay shared.
- `xreport_studio`: upload, preparation, dataset processing, tokenizer catalogue,
  validation statistics, training, evaluation, architecture presets, the Studio
  ORM schema and Alembic stream (`migrations/studio`).
- `xreport_inference`: inference runtime and catalogue, Hugging Face access,
  model installation/storage, report history, the Inference ORM schema and
  Alembic stream (`migrations/inference`) and `desktop_entry`.
- `xreport_studio` and `xreport_inference` never import each other and
  `xreport_shared` imports neither (`tests/unit/test_packaging_boundary.py` checks
  imports and module-name strings in every file). Applications announce their
  database schema to the shared package when imported
  (`register_application_schema`); the shared package never loads an
  application by name.
- Each application's `__init__` selects its data layout (`XREPORT_APP`, an
  explicit launcher value wins) before the shared package resolves runtime paths.
  One process hosts one application.
- Python dependencies are still one environment definition (`app/server/pyproject.toml`
  and `uv.lock`, extras `standard`/`cuda`/`studio`) with two virtual
  environments (`.venv`, `.venv-studio`). The Inference build never imports Studio
  code; per-application lock files are not implemented.

Frontend (`app/client`, one npm workspace, three Angular source projects):

- `projects/shared` (`@xreport/shared/*` alias): layout, settings page, theming,
  jobs/polling/status services, API request infrastructure, generated API types.
- `projects/studio`, `projects/inference`: entry point, routes, pages,
  components, services and proxy configuration of one application each. Angular
  `no-restricted-imports` rules and `tests/unit/test_frontend_boundary.py` fail
  on cross-application imports. Shared component styles are referenced by
  relative path because `styleUrl` cannot use the alias.
- `npm test` runs the `shared-ui`, `client-angular` and `studio-angular` unit
  test targets; one `package.json` still serves all three projects.

## Backend composition

Each application assembles only its own routers
(`xreport_*/app.py` → `xreport_shared.app_factory.create_application`).

- XREPORT Inference mounts inference, jobs, settings, database settings, and
  `POST /api/exchange/import`.
- Studio mounts upload, preparation, training, validation, tokenizers, model
  architecture presets, jobs, settings, database settings,
  `POST /api/exchange/export`, and `GET /api/exchange/exports/{filename}`.
- Each application has its own OpenAPI snapshot
  (`app/shared/openapi.xreport.json`, `openapi.studio.json`, regenerated with
  `app/scripts/export_openapi.py`) and generated client types.
- `TransformerEncoder`/`TransformerDecoder` serialise `ffn_units`, `dropout`
  and `key_dim`; `PositionalEmbedding`, the masked loss and the masked accuracy
  serialise `pad_token_id`. Configurations saved before these keys existed fall
  back to the original values, so older checkpoints load unchanged.
- Studio checkpoints store the resolved architecture in
  `configuration/configuration.json` (`architecture`, `architecture_preset`,
  and the measured `architecture_summary`). The manifest architecture id is
  unchanged because every configurable architecture keeps the BEiT encoder and
  the same layer types.

- Each profile has its own ORM schema, Alembic stream, version table, and
  settings row (`repositories/schemas/{inference,studio}.py`,
  `migrations/{inference,studio}`). A database is owned by exactly one
  application; the other application refuses to open or migrate it. Studio
  stores no inference settings. See [persistence](persistence.md).
- Python dependencies are one lock file. `nltk` and `openpyxl` (evaluation and
  spreadsheet upload) and `opencv-python`/`matplotlib` live in the `studio` extra,
  so the XREPORT Inference environment does not install them. `huggingface-hub`
  and the other inference packages remain in the base set because Studio
  validation runs models.

## Data ownership

- XREPORT Inference owns `<resources>/database.db` (or its configured
  PostgreSQL database), reports, installed models, and settings. Databases
  created with the retired shared schema (releases up to 3.3.1) are rejected
  and must be recreated; their history is not carried forward.
- Studio uses `<resources>/studio/` for its database, `database-config.json`,
  checkpoints, exports, and logs.
- The pinned BEiT encoder and tokenizers (`<resources>/models`) are shared,
  immutable resources installed atomically. Studio resolves `MODELS_DIR` to the
  base folder; it never writes elsewhere in XREPORT Inference's tree.
  Nothing serialises concurrent writers to the shared `models` folder, so two
  applications installing the same encoder or tokenizer at once is an accepted
  risk.
- A process lock (`.xreport.lock` in each data folder) makes a second process on
  the same folder fail at startup instead of mutating the same SQLite file.
  The OS releases it when the owner exits or crashes.
- Launcher cleanup is application-scoped (`-App Xreport|Studio`). Studio
  targets never include `models`. XREPORT Inference "remove all data" still removes the
  shared `models` folder (existing behaviour) and says so in its prompt; it
  never touches `studio/`.

## Checkpoint exchange (`.xreportckpt`, format version 2)

A zip with an allow-listed layout:

```
manifest.json
saved_model.keras
configuration/{configuration,metadata,session_history}.json
tokenizer/<files>          bundled tokenizer (saved with save_pretrained)
encoder/<files>            optional; only when exported with the encoder
```

The manifest (`xreport_shared/domain/checkpoint_exchange.py`) records the format and
version, architecture id `xreport-beit-transformer`, `custom_objects_api`,
checkpoint name, `content_sha256` (the checkpoint identity), per-file size and
SHA-256, producer versions (application, Keras, Torch), tokenizer name (plus revision, vocabulary size and the pad/start/end token ids, all mandatory), encoder
repository and pinned revision, and minimal provenance (creation time, epochs,
last losses). It contains no dataset records, paths, or credentials.

The tokenizer vocabulary size and pad/start/end ids are mandatory: export refuses
a checkpoint whose metadata does not record them, import rejects a manifest that
omits them or disagrees with `metadata.json`, and the runtime never substitutes
the tokenizer's current ids. Checkpoints trained without them must be retrained.

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
   object compatibility, the encoder repository and pinned revision (a different
   encoder is rejected), name conflicts. Invalid input returns 400/409 before
   any job starts.
2. Streaming extraction to `checkpoints/.import-<id>/` with the byte count
   enforced against the manifest, then SHA-256 verification of every file and a
   check that `metadata.json` defines an integer `max_report_size` and a
   `tokenizer` equal to the manifest's tokenizer name.
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
Imports require a version 2 manifest. Version 1 and manifestless archives are
rejected; a lone `.keras` file is rejected with an explanation of the required
companion files.

Export (Studio) validates the registered artifact, refuses archives the import
limits would reject (member names, entry count, entry and total size), collects the tokenizer
through `TokenizerHandler`, hashes all members, and writes
`<resources>/studio/exports/<name>.xreportckpt`, offered as a download.
Training and export jobs exclude each other atomically. Checkpoint deletion
checks both job types while holding the same job-registration lock.

## Not covered

Image preprocessing (224x224 input and normalisation) is fixed in code on both
sides and is not part of the manifest. Import from arbitrary third-party frameworks, a live link between the two
applications, and CUDA-specific package acceptance are out of scope or
unverified; see the [status ledger](../project_status_ledger.md).
