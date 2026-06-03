# Runtime Deployment

Last updated: 2026-06-03

## Packaging Scope

- Current deployment and distribution implementation is Windows-focused.
- Installer and portable artifacts are exported to `release/windows`.

## Build Flow

1. Prepare runtimes through the launcher.
2. Build the Tauri app.
3. Export Windows artifacts with `tauri:export:windows`.

## Packaging Prerequisites

- Rust and Cargo must be available on the build machine.
- Runtime assets should be prepared before packaging.

## Runtime Lock Consistency

- Packaged workflow consistency is anchored to `runtimes/uv.lock`.
