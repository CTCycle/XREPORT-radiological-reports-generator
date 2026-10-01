# Clean-SHA release rebinding — 2026-10-01

Status: `SUPERSEDED` historical attempt record. The failed intermediate
rebinding described below was followed by a successful official CPU/CUDA
rebuild and verifier run from source commit
`301a5a510e494c501d6c25eb22d5baea824e936a`, with `dirty_tree=false` for both
variants. Release approval remains `PARTIAL` only for the genuine no-GPU,
hosted-CI, S51, and approval-manifest boundaries described in the current
[release-validation record](summary-20260930.md).

The superseding clean release pair records runtime payload hashes
`03da18158b95d0e8e7a400f76b4a151f189bce3fd29a2eb69fa8d7fb012631d4` (CPU)
and `d69ee87c7f627c277e68e21fcdb49340cb702ac8e2aff402e776a8445ac02f3f`
(CUDA). Both official artifact verifiers passed; the current release-folder
portable/MSI pairs, checksum files, and build metadata are present.

The earlier clean-source report is retained as build-history context: it
reported CPU and CUDA v3.1.0 metadata with `dirty_tree=false`, payload hashes
`e06e0be4fa22427a4e0cb6950c97cbe84f6f468b6fb625a0e909c7b47fc65a21` (CPU) and
`99ec44eb2333d573330abe454d85b181acd363dfdd33653dc01f15417b537448` (CUDA),
and successful artifact verification. The current `release/` folder retains
the CPU pair only; the historical CUDA files are not available for an
independent current rebind.

The failed attempt described here used implementation revision
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

At the time of this failed attempt, the CPU build remained bound to
`af666df53b3424c08a279bc9d0b79b81078815bd` with `dirty_tree=false`; the
successful clean pair described at the top supersedes that state and closes
current CPU/CUDA artifact binding.

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
