# Changelog

## 4.0.0 (2026-10-28)

Major release marking the split of XREPORT into two applications that are developed, configured and shipped separately. Releases up to 3.3.1 shipped one combined application.

### Distribution

- **Only XREPORT Inference is packaged.** The Windows desktop packages (CPU and CUDA portable executable and MSI, with checksums and build metadata) contain XREPORT Inference only. The packaged backend collects `xreport_inference` and `xreport_shared`; `xreport_studio`, its migrations and its Angular application are excluded, and the runtime audit rejects Studio members.
- **XREPORT Studio is not distributed.** It runs from the repository (`start_on_windows.ps1 -Action LaunchStudio`) with its own environment and data folder. Installing the Inference package does not install or launch Studio.

### Compatibility changes

- Inference and Studio use separate database schemas, Alembic streams and data folders. Each application refuses the other's database and the retired shared schema without modifying it. There is no conversion: a database created by 3.3.1 or earlier must be replaced (see [persistence](assets/docs/architecture/persistence.md)).
- Models move from Studio to Inference as `.xreportckpt` archives; archives without a manifest and format 1 checkpoints are rejected.
- The Data access setting and filesystem gate were removed from both applications.
- Package-internal module layout changed to `xreport_shared`, `xreport_inference` and `xreport_studio`; the Angular code is split into `inference`, `studio` and `shared` projects.

### Inference

- Per-image batch runs with a report selector, independent editing and saving per report, and generation tracking that survives navigation.
- An independent study per model, a Free memory status-bar action and an idle-unload setting for resident models.
- Configurable storage root with a validated copy-verify-switch move, user-adjustable generation limits, Hugging Face token settings, update-check progress and gated-download errors.
- System status bar and notification centre; edited words are highlighted against the original output on report detail.

### Studio

- Model page with an interactive architecture editor and presets; training from a selected preset; configurable FFN width, dropout and key dimension.
- Tokenizer catalogue (including DeBERTa v3) with token statistics in dataset validation, stored in relational tables.
- Gradient accumulation; training honours the selected device, and stopped runs keep a resumable checkpoint.
- Large checkpoints (above about 2 GB) save without the in-memory h5py limit; datasets up to 2 GB can be uploaded with progress shown.

### Fixes

- Training on the CPU choice no longer runs on CUDA; mixed precision and Windows dataloader workers no longer crash.
- A duplicate checkpoint name no longer overwrites an existing checkpoint.
- Job polling responses that outlast the poll interval are kept.
- The pinned BEiT revision lives in shared code, so a packaged Inference build can load XREPORT checkpoints.
- The Windows launcher's Studio environment probe survives Windows PowerShell 5.1.

### Known limitations

See the [status ledger](assets/docs/project_status_ledger.md) and the [validation ledger](assets/docs/validation_campaign_ledger.md). In short: PostgreSQL for the separated schemas, MedGemma (gated access), packaged checkpoint import, and Studio training of the default preset on the 6 GB reference GPU were not verified. The packages are unsigned and need WebView2.

Earlier releases are described on the [GitHub Releases](https://github.com/CTCycle/XREPORT-radiological-reports-generator/releases) page.
