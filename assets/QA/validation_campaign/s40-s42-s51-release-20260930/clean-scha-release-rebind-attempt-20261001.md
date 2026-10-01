# Clean-SHA release rebinding — 2026-10-01

Status: `PARTIAL` for current release rebinding. The CPU artifact pair was
constructed and verified from the clean source revision; the retained clean
CUDA report is historical and the current CUDA diagnostic output is
dirty-tree-bound. Release approval remains `PARTIAL` for current CUDA
artifacts, no-GPU, hosted-CI, and approval-manifest boundaries.

The earlier clean-source report is retained as build-history context: it
reported CPU and CUDA v3.1.0 metadata with `dirty_tree=false`, payload hashes
`e06e0be4fa22427a4e0cb6950c97cbe84f6f468b6fb625a0e909c7b47fc65a21` (CPU) and
`99ec44eb2333d573330abe454d85b181acd363dfdd33653dc01f15417b537448` (CUDA),
and successful artifact verification. The current `release/` folder retains
the CPU pair only; the historical CUDA files are not available for an
independent current rebind.

The current committed implementation revision is
`af666df53b3424c08a279bc9d0b79b81078815bd`. The official CPU artifact verifier
ran against the current CPU portable/MSI pair and reached its final QA-report
write after the artifact, runtime, checksum, portable, and MSI checks. The
child PowerShell process could not persist that report directly under
`assets/QA/desktop`; the current CPU verification receipt was refreshed through
the workspace patch path with the same committed source revision.

The official CUDA build was first attempted with
`app/desktop/build/tauri_build.ps1 -DesktopRuntime Cuda -DesktopTarget All
-Version 3.1.0 -AllowDirtyTree`. It stopped before backend freeze or Tauri
packaging while removing the generated
`app/desktop/src-tauri/ui` tree: all 38 generated items were protected by
host ACLs and `Remove-LauncherPath` correctly aborted the build. No CUDA
portable/MSI artifact was emitted or overwritten by this attempt.

An administrator-capable retry cleared that UI ACL boundary, but the official
launcher then exposed a PowerShell/native-argument quoting defect in the
working-tree PyInstaller probe. The probe was reduced to a quote-free import
and version check, and the exact probe then exited `0` with PyInstaller
`6.22.2`. The next build reached PyInstaller analysis but found the local
`pyinstaller-hooks-contrib 2026.6` installation missing its pinned
`pre_safe_import_module/hook-win32com.py`; a stale XREPORT pytest/Playwright
process was also holding the local environment's `playwright/driver/node.exe`.
That stale test process was stopped, the frozen desktop sync completed, and
the hook was restored and directly verified.

The earlier administrator-capable CUDA attempt completed the PyInstaller freeze,
created and verified the CUDA runtime bundle, and wrote
`assets/QA/desktop/runtime-cuda-3.1.0.json` with source revision
`af666df53b3424c08a279bc9d0b79b81078815bd`, `dirty_tree=true`, 6,789 files,
and payload SHA-256
`2d6fc6086a5662cf29c8276a6fa8d063ac10edbc2274d851150a1ec05d7dffbe`.
Tauri/Cargo then terminated through the `npm.cmd` wrapper with exit code `-1`
while compiling the desktop bundle. A later direct diagnostic with the corrected
`XREPORT_DESKTOP_VARIANT=cuda` environment completed Cargo release compilation
and reached WiX MSI bundling; it emitted the target-folder CUDA MSI and raw
desktop executable, but did not complete release post-processing, portable
packaging, checksum generation, or metadata publication. The emitted runtime
remained `dirty_tree=true` with the hash above and was not promoted to
`release/`. A read-only Windows Installer table check confirmed that the
diagnostic MSI contains `runtime.zip`; this confirms packaging composition only,
not clean provenance or release approval.

The current CPU build remains bound to
`af666df53b3424c08a279bc9d0b79b81078815bd` with `dirty_tree=false`; its
portable/MSI pair and verifier are current. The retained historical CUDA
metadata and verifier receipt refer to outputs no longer present in
`release/`, so they do not close current CUDA release binding.

The official frontend rebuild passed during each retry and the generated
`app/desktop/src-tauri/ui` staging tree remains present. The quote-safe probe
repair and the small `WaitForExit` output-suppression repair are included in
the current review set.

After cleanup there is no XREPORT process or listener on the validation ports.
The existing CPU/CUDA native receipts and S51 receipts retain their recorded
earlier provenance; they were not relabeled as current clean-SHA evidence. The
clean packaged CUDA Generate result is recorded in
[packaged-cuda-inference-clean-20261001.md](packaged-cuda-inference-clean-20261001.md)
as historical technical evidence. MSI install and uninstall lifecycle testing
was not performed. No validation gate is marked failed solely because the
earlier build attempt ended with the wrapper exit `-1`; the current CUDA
release gap is recorded as an evidence boundary instead.
