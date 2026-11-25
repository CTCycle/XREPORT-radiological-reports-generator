# Changelog

## 4.0.1 (2026-11-25)

Patch release. XREPORT Inference is still the only packaged application; databases, checkpoints and settings from 4.0.0 are used unchanged (no schema migration).

### Fixes

- **Copied and exported drafts carry their research status.** Inference's Copy and Export text now begin with a research-use-only notice, the model and revision, the source image name(s) and the export time. A combined study report lists every image read and exports as `xreport-study-<first image>-<n>-images.txt` instead of being named after the first image alone (`ISSUE-006`, part of `ISSUE-008`).
- The tagged desktop release workflow calls `setup_and_maintenance.ps1`; it still referenced the renamed `start_on_windows.ps1` and could not build packages.
- The desktop window no longer enforces a 1024×720 minimum size and the shared layout no longer sets a minimum body width; the guided tour follows the actual viewport.

### Windows launcher and maintenance

- `start_on_windows.bat` and `start_studio_on_windows.bat` only start the applications. Dependency setup, desktop builds, cleanup and uninstall moved to `setup_and_maintenance.ps1` (menu or `-Action`).
- Cleanup and uninstall validate every target root, support `-DryRun`, preserve persistent user data and shared settings, leave reparse points untouched, continue past independent failures and uninstall the XREPORT Inference CPU/CUDA MSI products. Counting and deletion use a streaming traversal and are much faster on large caches.
- Studio environment setup retries a failed runtime probe once by reinstalling only the locked Keras package when the probe reports a missing Keras module.

### Internal

- Backend layering, typing and complexity clean-up with no API or behaviour change; Pyright reports no errors (the local configuration now also excludes the Studio virtual environment).

### Known limitations

Open issues are tracked in the [status ledger](assets/docs/project_status_ledger.md): unsuitable inputs are not yet flagged before generation (rest of `ISSUE-006`), the Reports card of a combined study still shows only the first image (rest of `ISSUE-008`), Studio training fails in an environment with an incomplete Keras install (`ISSUE-007`), dataset split errors and browser-mode folder selection need work (`ISSUE-009`, `ISSUE-010`), and MedGemma needs gated access (`ISSUE-002`). The packages are unsigned and need WebView2.

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
