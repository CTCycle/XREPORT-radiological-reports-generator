# Launcher cleanup performance

Date: 2026-11-04
Branch: `develop`
Baseline source: `d4191643` (`app/scripts/launcher_cleanup.psm1` before the working-tree optimization)

## Isolated filesystem benchmark

The baseline module was loaded from the starting `HEAD`; the optimized module was loaded from the working tree. Each module received a separate, equivalent temporary fixture containing 32 groups, three nested directories per group, 32 files per leaf directory (1,024 files total), and one root `.gitkeep`. Fixture creation was excluded from timings. Preview/counting includes `New-XReportCleanupPlan` and a read-only `Invoke-XReportCleanupPlan -DryRun`. Deletion was timed separately after the preview. Both implementations counted and removed 1,120 entries (1,024 files and 96 directories), leaving `.gitkeep` in place.

| Engine | Plan construction | Preview/counting | Deletion | Combined |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2.04 s | 2.14 s | 80.38 s | 82.53 s |
| Optimized | 0.04 s | 0.60 s | 1.17 s | 1.77 s |

Combined filesystem time was 46.6× lower in this single run. Preview/counting was 3.6× faster and deletion was 68.7× faster. Fixture generation and cleanup were outside the measured intervals. These timings are evidence for this host and fixture, not an automated timing threshold.

## Launcher dry-run checks

All commands below exited successfully and remained read-only on the real checkout/profile:

| Action | Elapsed |
| --- | ---: |
| `ClearCache -DryRun` | 22.70 s; 37,281 entries |
| `RemoveAllData -App Xreport -DryRun` | 1.04 s; 8 entries |
| `Uninstall -DryRun` | 48.44 s; 106,096 entries |

All three runs reported zero failures and made no changes. An earlier `ClearCache -DryRun` observation was 132.78 s. Real-tree timing varies with cache state, so the isolated fixture above is the controlled before/after result. The uninstall dry run only discovered and displayed MSI products; no MSI action ran.

## Regression coverage and limits

`app/tests/test_launcher_cleanup.ps1` passed with large-tree count/deletion checks, dry-run immutability, sentinel preservation, grouped default output, verbose path output, strict handling of a locked file, continued cleanup of an independent sibling, out-of-root rejection, per-app data boundaries, target exclusion beneath covered uninstall roots, and mocked MSI selection/uninstall. The fixture host denied junction creation, so junction traversal was not exercised in this run. No repository/profile cleanup or real MSI uninstall was performed.
