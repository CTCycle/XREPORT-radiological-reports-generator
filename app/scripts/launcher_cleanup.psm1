Set-StrictMode -Version 2.0

function ConvertTo-CleanupFullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { throw 'Cleanup path cannot be empty.' }
    $fullPath = [IO.Path]::GetFullPath($Path)
    $root = [IO.Path]::GetPathRoot($fullPath)
    if ([string]::IsNullOrWhiteSpace($root) -or $fullPath.TrimEnd('\', '/') -eq $root.TrimEnd('\', '/')) {
        throw "Refusing to target a filesystem root: $fullPath"
    }
    return $fullPath.TrimEnd('\', '/')
}

function Test-CleanupPathUnderRoot {
    param([string]$Path, [string]$Root)
    $normalizedRoot = $Root.TrimEnd('\', '/')
    $normalizedPath = $Path.TrimEnd('\', '/')
    return $normalizedPath.Equals($normalizedRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $normalizedPath.StartsWith($normalizedRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        $normalizedPath.StartsWith($normalizedRoot + [IO.Path]::AltDirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}

function Resolve-XReportStoragePointer {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$BootstrapRoot)
    $pointerPath = Join-Path $BootstrapRoot 'storage-location.json'
    if (-not (Test-Path -LiteralPath $pointerPath -PathType Leaf)) { return $null }
    $pointer = Get-Content -LiteralPath $pointerPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    $value = if ($pointer -and $pointer.storage_root -is [string]) { [string]$pointer.storage_root } else { '' }
    if ([string]::IsNullOrWhiteSpace($value)) { return $null }
    $value = [Environment]::ExpandEnvironmentVariables($value.Trim())
    if (-not [IO.Path]::IsPathRooted($value)) { $value = Join-Path $BootstrapRoot $value }
    return [IO.Path]::GetFullPath($value)
}

function Get-XReportApplicationDataTargets {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string[]]$ResourceRoots,
        [ValidateSet('Xreport', 'Studio')][string]$App = 'Xreport',
        [switch]$CheckpointsOnly
    )
    $targets = [Collections.Generic.List[object]]::new()
    $appLabel = if ($App -eq 'Studio') { 'XREPORT Studio' } else { 'XREPORT Inference' }
    foreach ($resourceRoot in @($ResourceRoots | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
        $appRoot = if ($App -eq 'Studio') { Join-Path $resourceRoot 'studio' } else { $resourceRoot }
        if ($CheckpointsOnly) {
            [void]$targets.Add([pscustomobject]@{ Path = (Join-Path $appRoot 'checkpoints'); Label = "$appLabel checkpoints"; KeepRoot = $true; PreserveNames = @('.gitkeep') })
            continue
        }
        $databasePath = Join-Path $appRoot 'database.db'
        foreach ($path in @($databasePath, "$databasePath-wal", "$databasePath-shm", "$databasePath-journal")) {
            [void]$targets.Add([pscustomobject]@{ Path = $path; Label = "$appLabel database"; KeepRoot = $false; PreserveNames = @() })
        }
        foreach ($name in @('checkpoints', 'exports', 'logs')) {
            [void]$targets.Add([pscustomobject]@{ Path = (Join-Path $appRoot $name); Label = "$appLabel $name"; KeepRoot = $true; PreserveNames = @('.gitkeep') })
        }
        if ($App -ne 'Studio') {
            foreach ($name in @('models', 'templates', 'validation_receipts')) {
                $label = if ($name -eq 'models') { 'Shared XREPORT Inference and Studio models' } else { "XREPORT Inference $name" }
                [void]$targets.Add([pscustomobject]@{ Path = (Join-Path $resourceRoot $name); Label = $label; KeepRoot = $true; PreserveNames = @('.gitkeep') })
            }
        }
    }
    $unique = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($target in $targets) {
        $key = [IO.Path]::GetFullPath([string]$target.Path)
        if (-not $unique.ContainsKey($key)) { $unique.Add($key, $target) }
    }
    return @($unique.Values)
}

function Test-CleanupReparseChain {
    param([string]$Path, [string]$AllowedRoot)
    $cursor = $Path
    while (-not [string]::IsNullOrWhiteSpace($cursor)) {
        if (Test-Path -LiteralPath $cursor) {
            try {
                $item = Get-Item -LiteralPath $cursor -Force -ErrorAction Stop
                if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { return $false }
            } catch { return $false }
        }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) { break }
        $cursor = $parent
    }
    return $true
}

function New-XReportCleanupPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][object[]]$Targets,
        [Parameter(Mandatory = $true)][string[]]$AllowedRoots
    )

    $roots = @($AllowedRoots | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\', '/') } | Select-Object -Unique)
    if ($roots.Count -eq 0) { throw 'At least one validated cleanup root is required.' }
    $items = [Collections.Generic.List[object]]::new()
    foreach ($spec in $Targets) {
        $pathValue = if ($spec -is [string]) { [string]$spec } else { [string]$spec.Path }
        $label = if ($spec -is [string]) { 'Application cleanup' } elseif ($spec.Label) { [string]$spec.Label } else { 'Application cleanup' }
        $keepRoot = if ($spec -is [string]) { $false } else { [bool]$spec.KeepRoot }
        $preserveNames = if ($spec -is [string] -or -not $spec.PreserveNames) { @() } else { @($spec.PreserveNames | ForEach-Object { [string]$_ }) }
        $planError = if ($spec -isnot [string] -and $spec.PSObject.Properties['PlanError']) { [string]$spec.PlanError } else { '' }
        try { $target = ConvertTo-CleanupFullPath -Path $pathValue }
        catch {
            [void]$items.Add([pscustomobject]@{ Path = $pathValue; Label = $label; Kind = 'Target'; Status = 'Failed'; Message = $_.Exception.Message })
            continue
        }
        $allowedRoot = $null
        foreach ($root in $roots) {
            if ((Test-CleanupPathUnderRoot -Path $target -Root $root) -and -not $target.TrimEnd('\', '/').Equals($root.TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase)) { $allowedRoot = $root; break }
        }
        if (-not $allowedRoot) {
            [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; Kind = 'Target'; Status = 'Failed'; Message = 'Target is outside every allowlisted application root.' })
            continue
        }
        if (-not (Test-CleanupReparseChain -Path $target -AllowedRoot $allowedRoot)) {
            [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; Kind = 'Target'; Status = 'Skipped'; Message = 'A reparse point exists in the target path; it was not traversed.' })
            continue
        }
        if (-not [string]::IsNullOrWhiteSpace($planError)) {
            [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; Kind = 'Target'; Status = 'Failed'; Message = $planError })
            continue
        }
        try { $rootItem = Get-Item -LiteralPath $target -Force -ErrorAction Stop }
        catch {
            if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound) {
                [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; Kind = 'Target'; Status = 'Skipped'; Message = 'Target does not exist.' })
            } else {
                [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; Kind = 'Target'; Status = 'Failed'; Message = $_.Exception.Message })
            }
            continue
        }

        $stack = [Collections.Generic.Stack[object]]::new()
        $stack.Push([pscustomobject]@{ Item = $rootItem; IsRoot = $true })
        $preserved = [Collections.Generic.List[string]]::new()
        while ($stack.Count -gt 0) {
            $frame = $stack.Pop()
            $entry = $frame.Item
            $isRoot = [bool]$frame.IsRoot
            $entryPath = [IO.Path]::GetFullPath($entry.FullName).TrimEnd('\', '/')
            if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                [void]$items.Add([pscustomobject]@{ Path = $entryPath; Label = $label; Kind = 'Entry'; Status = 'Skipped'; Message = 'Reparse point left untouched.' })
                continue
            }
            if (-not $isRoot -and $preserveNames -contains [string]$entry.Name) {
                [void]$preserved.Add($entryPath)
                [void]$items.Add([pscustomobject]@{ Path = $entryPath; Label = $label; Kind = 'Entry'; Status = 'Preserved'; Message = 'Preserve-name rule.' })
                continue
            }
            if ($entry.PSIsContainer) {
                try {
                    $children = @(Get-ChildItem -LiteralPath $entryPath -Force -ErrorAction Stop)
                    foreach ($child in $children) { $stack.Push([pscustomobject]@{ Item = $child; IsRoot = $false }) }
                } catch {
                    [void]$items.Add([pscustomobject]@{ Path = $entryPath; Label = $label; Kind = 'Entry'; Status = 'Failed'; Message = "Cannot enumerate directory: $($_.Exception.Message)" })
                }
                if (-not ($isRoot -and $keepRoot)) {
                    [void]$items.Add([pscustomobject]@{ Path = $entryPath; Label = $label; Kind = 'Directory'; Status = 'Planned'; Message = '' })
                }
            } else {
                [void]$items.Add([pscustomobject]@{ Path = $entryPath; Label = $label; Kind = 'File'; Status = 'Planned'; Message = '' })
            }
        }
    }

    $protectedPaths = @($items | Where-Object { $_.Status -in @('Preserved', 'Skipped', 'Failed') } | Select-Object -ExpandProperty Path)
    $planned = @($items | Where-Object {
        if ($_.Status -ne 'Planned') { return $false }
        if ($_.Kind -ne 'Directory') { return $true }
        foreach ($protectedPath in $protectedPaths) {
            if (Test-CleanupPathUnderRoot -Path $protectedPath -Root $_.Path) { return $false }
        }
        return $true
    } | Sort-Object @{ Expression = { $_.Path.Length }; Descending = $true }, @{ Expression = { $_.Path.ToUpperInvariant() }; Descending = $false })
    $initial = @($items | Where-Object Status -ne 'Planned')
    return [pscustomobject]@{ Items = $planned; InitialResults = $initial; AllowedRoots = $roots; CreatedAt = [DateTime]::UtcNow }
}

function Invoke-XReportCleanupPlan {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][pscustomobject]$Plan, [switch]$DryRun)

    $results = [Collections.Generic.List[object]]::new()
    foreach ($item in $Plan.InitialResults) { [void]$results.Add($item) }
    foreach ($item in $Plan.Items) {
        if ($DryRun) {
            [void]$results.Add([pscustomobject]@{ Path = $item.Path; Label = $item.Label; Kind = $item.Kind; Status = 'Planned'; Message = '' })
            continue
        }
        $root = $null
        foreach ($candidate in $Plan.AllowedRoots) {
            if (Test-CleanupPathUnderRoot -Path $item.Path -Root $candidate) { $root = $candidate; break }
        }
        if (-not $root -or -not (Test-CleanupReparseChain -Path $item.Path -AllowedRoot $root)) {
            [void]$results.Add([pscustomobject]@{ Path = $item.Path; Label = $item.Label; Kind = $item.Kind; Status = 'Skipped'; Message = 'Path changed or became a reparse point after planning.' })
            continue
        }
        try {
            $current = Get-Item -LiteralPath $item.Path -Force -ErrorAction Stop
            if (($current.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Entry became a reparse point after planning.' }
            if ($current.PSIsContainer -and @(Get-ChildItem -LiteralPath $item.Path -Force -ErrorAction Stop).Count -gt 0) {
                throw 'Directory changed after planning; new contents were left untouched.'
            }
            Remove-Item -LiteralPath $item.Path -Force -ErrorAction Stop
            [void]$results.Add([pscustomobject]@{ Path = $item.Path; Label = $item.Label; Kind = $item.Kind; Status = 'Cleaned'; Message = '' })
        } catch {
            [void]$results.Add([pscustomobject]@{ Path = $item.Path; Label = $item.Label; Kind = $item.Kind; Status = 'Failed'; Message = $_.Exception.Message })
        }
    }
    return [pscustomobject]@{
        DryRun = [bool]$DryRun
        Results = @($results)
        Planned = @($results | Where-Object Status -eq 'Planned')
        Cleaned = @($results | Where-Object Status -eq 'Cleaned')
        Skipped = @($results | Where-Object Status -eq 'Skipped')
        Failed = @($results | Where-Object Status -eq 'Failed')
        Preserved = @($results | Where-Object Status -eq 'Preserved')
    }
}

function Get-XReportCacheTargets {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [string]$LocalAppData,
        [string[]]$ResourceRoots = @()
    )
    $targets = [Collections.Generic.List[object]]::new()
    $add = { param([string]$Path, [string]$Label, [switch]$KeepRoot, [string]$PlanError)
        if (-not [string]::IsNullOrWhiteSpace($Path)) { [void]$targets.Add([pscustomobject]@{ Path = $Path; Label = $Label; KeepRoot = [bool]$KeepRoot; PreserveNames = @('.gitkeep'); PlanError = $PlanError }) }
    }
    & $add (Join-Path $RepoRoot 'runtimes\cache') 'Repository runtime and tool caches' -KeepRoot
    & $add (Join-Path $RepoRoot 'app\server\runtimes\cache') 'Legacy backend cache root'
    & $add (Join-Path $RepoRoot 'app\client\.angular\cache') 'Legacy Angular CLI cache'
    & $add (Join-Path $RepoRoot 'app\client\node_modules\.cache') 'Frontend dependency cache'
    & $add (Join-Path $RepoRoot 'app\client\node_modules\.vite') 'Frontend Vite transform cache'
    & $add (Join-Path $RepoRoot 'app\client\dist') 'Generated Angular frontend'
    & $add (Join-Path $RepoRoot 'app\client\dist-ssr') 'Generated Angular SSR output'
    & $add (Join-Path $RepoRoot 'app\client\out-tsc') 'Generated TypeScript test output'
    & $add (Join-Path $RepoRoot 'app\client\coverage') 'Generated frontend coverage output'
    & $add (Join-Path $RepoRoot 'htmlcov') 'Python coverage HTML output'
    & $add (Join-Path $RepoRoot '.coverage') 'Python coverage database'
    try {
        foreach ($coverageFile in Get-ChildItem -LiteralPath $RepoRoot -File -Force -ErrorAction Stop | Where-Object { $_.Name -match '^\.coverage\.' }) {
            & $add $coverageFile.FullName 'Parallel Python coverage database'
        }
    } catch { }
    foreach ($name in @('.pytest_cache', '.ruff_cache', '.mypy_cache', '__pycache__')) {
        & $add (Join-Path $RepoRoot $name) 'Repository-root Python/tool cache'
    }
    & $add (Join-Path $RepoRoot 'app\client\.cache') 'Frontend compiler cache'
    if (-not [string]::IsNullOrWhiteSpace($LocalAppData)) {
        $appData = Join-Path $LocalAppData 'XREPORT\data'
        & $add (Join-Path $appData 'runtimes\cache') 'Packaged runtime and tool caches' -KeepRoot
        & $add (Join-Path $appData 'nltk') 'Packaged NLTK data'
        $runtimeRoot = Join-Path $appData 'runtimes'
        if (Test-Path -LiteralPath $runtimeRoot -PathType Container) {
            if (-not (Test-CleanupReparseChain -Path $runtimeRoot -AllowedRoot $appData)) {
                & $add $runtimeRoot 'Packaged runtime staging scan skipped'
            } else {
                try {
                    foreach ($staging in Get-ChildItem -LiteralPath $runtimeRoot -Directory -Force -ErrorAction Stop | Where-Object { $_.Name -match '^\.staging-[0-9a-fA-F]{32}$' }) {
                        & $add $staging.FullName 'Incomplete packaged runtime extraction'
                    }
                } catch { & $add $runtimeRoot 'Packaged runtime staging scan failed' -PlanError $_.Exception.Message }
            }
        }
    }

    foreach ($path in @(
        (Join-Path $RepoRoot 'app\desktop\node_modules\.cache'),
        (Join-Path $RepoRoot 'app\desktop\node_modules\.vite'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\target'),
        (Join-Path $RepoRoot 'app\desktop\build\__pycache__'),
        (Join-Path $RepoRoot 'app\desktop\build\runtime-staging'),
        (Join-Path $RepoRoot 'app\desktop\build\pyinstaller'),
        (Join-Path $RepoRoot 'app\desktop\build\cpu-overlay'),
        (Join-Path $RepoRoot 'app\desktop\build\cargo-target'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\generated'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\gen'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\app'),
        (Join-Path $RepoRoot 'app\desktop\src-tauri\ui')
    )) { & $add $path 'Generated desktop build state' }
    $qaCacheRoot = Join-Path $RepoRoot 'app\server\assets\QA'
    if (Test-Path -LiteralPath $qaCacheRoot -PathType Container) {
        if (-not (Test-CleanupReparseChain -Path $qaCacheRoot -AllowedRoot $RepoRoot)) {
            & $add $qaCacheRoot 'QA cache scan skipped'
        } else {
            try {
                foreach ($qaCache in Get-ChildItem -LiteralPath $qaCacheRoot -Directory -Force -ErrorAction Stop | Where-Object { $_.Name -match '^pytest-cache-[A-Za-z0-9._-]+$' }) {
                    & $add $qaCache.FullName 'Ignored disposable pytest cache'
                }
            } catch { & $add $qaCacheRoot 'QA cache scan failed' -PlanError $_.Exception.Message }
        }
    }
    $desktopBuild = Join-Path $RepoRoot 'app\desktop\build'
    if (Test-Path -LiteralPath $desktopBuild -PathType Container) {
        if (-not (Test-CleanupReparseChain -Path $desktopBuild -AllowedRoot $RepoRoot)) {
            & $add $desktopBuild 'Generated desktop configuration scan skipped'
        } else {
            try {
                foreach ($config in Get-ChildItem -LiteralPath $desktopBuild -File -Filter 'tauri-*.json' -Force -ErrorAction Stop) {
                    & $add $config.FullName 'Generated Tauri build configuration'
                }
            } catch { & $add $desktopBuild 'Generated desktop configuration scan failed' -PlanError $_.Exception.Message }
        }
    }
    $launcherRuntimeDir = Join-Path $RepoRoot 'runtimes'
    if (Test-Path -LiteralPath $launcherRuntimeDir -PathType Container) {
        if (-not (Test-CleanupReparseChain -Path $launcherRuntimeDir -AllowedRoot $RepoRoot)) {
            & $add $launcherRuntimeDir 'Launcher runtime-state scan skipped'
        } else {
            try {
                foreach ($temporaryState in Get-ChildItem -LiteralPath $launcherRuntimeDir -File -Force -ErrorAction Stop | Where-Object { $_.Name -match '^studio-install-state\.json\.\d+\.tmp$' }) {
                    & $add $temporaryState.FullName 'Stale launcher state temporary'
                }
            } catch { & $add $launcherRuntimeDir 'Launcher runtime-state scan failed' -PlanError $_.Exception.Message }
        }
    }

    $scanRoots = @(
        (Join-Path $RepoRoot 'app\client'),
        (Join-Path $RepoRoot 'app\server'),
        (Join-Path $RepoRoot 'app\scripts'),
        (Join-Path $RepoRoot 'app\tests'),
        (Join-Path $RepoRoot 'app\xreport_shared'),
        (Join-Path $RepoRoot 'app\xreport_inference'),
        (Join-Path $RepoRoot 'app\xreport_studio'),
        (Join-Path $RepoRoot 'app\desktop\build')
    )
    $cacheNames = @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.cache', '.vite', '.angular-cache', 'htmlcov', 'coverage', 'out-tsc')
    $pruneNames = @('.git', '.venv', '.venv-studio', 'node_modules', 'dist', 'dist-ssr', 'target', 'runtimes', 'release')
    foreach ($scanRoot in $scanRoots) {
        if (-not (Test-Path -LiteralPath $scanRoot -PathType Container)) { continue }
        if (-not (Test-CleanupReparseChain -Path $scanRoot -AllowedRoot $RepoRoot)) { & $add $scanRoot 'Nested cache scan skipped'; continue }
        $stack = [Collections.Generic.Stack[string]]::new()
        $stack.Push([IO.Path]::GetFullPath($scanRoot))
        while ($stack.Count -gt 0) {
            $directory = $stack.Pop()
            try { $entries = @(Get-ChildItem -LiteralPath $directory -Force -ErrorAction Stop) } catch { & $add $directory 'Nested cache scan failed' -PlanError $_.Exception.Message; continue }
            foreach ($entry in $entries) {
                if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    if ($entry.PSIsContainer -and $cacheNames -contains $entry.Name) { & $add $entry.FullName 'Nested Python/tool cache reparse point' }
                    continue
                }
                if ($entry.PSIsContainer) {
                    if ($entry.Name -match '^pytest-cache-[A-Za-z0-9._-]+$' -and $directory.Equals($qaCacheRoot, [StringComparison]::OrdinalIgnoreCase)) { & $add $entry.FullName 'Ignored disposable pytest cache'; continue }
                    if ($cacheNames -contains $entry.Name) { & $add $entry.FullName 'Nested Python/tool cache'; continue }
                    if ($pruneNames -contains $entry.Name) { continue }
                    $stack.Push($entry.FullName)
                } elseif ($entry.Extension -in @('.pyc', '.pyo') -or $entry.Name -eq '.coverage' -or $entry.Name -match '^\.coverage\.') {
                    & $add $entry.FullName 'Python bytecode or coverage artifact'
                }
            }
        }
    }

    $bytecodeRoots = @(
        (Join-Path $RepoRoot 'app\server\.venv'),
        (Join-Path $RepoRoot 'app\server\.venv-studio'),
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $RepoRoot 'runtimes\python')
    )
    foreach ($bytecodeRoot in $bytecodeRoots) {
        if (-not (Test-Path -LiteralPath $bytecodeRoot -PathType Container)) { continue }
        if (-not (Test-CleanupReparseChain -Path $bytecodeRoot -AllowedRoot $RepoRoot)) { & $add $bytecodeRoot 'Python bytecode scan skipped'; continue }
        $environmentCacheNames = @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.cache')
        $stack = [Collections.Generic.Stack[string]]::new(); $stack.Push([IO.Path]::GetFullPath($bytecodeRoot))
        while ($stack.Count -gt 0) {
            $directory = $stack.Pop()
            try { $entries = @(Get-ChildItem -LiteralPath $directory -Force -ErrorAction Stop) } catch { & $add $directory 'Python environment cache scan failed' -PlanError $_.Exception.Message; continue }
            foreach ($entry in $entries) {
                if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    if ($entry.PSIsContainer -and $entry.Name -eq '__pycache__') { & $add $entry.FullName 'Python environment bytecode reparse point' }
                    continue
                }
                if ($entry.PSIsContainer) {
                    if ($environmentCacheNames -contains $entry.Name) { & $add $entry.FullName 'Python environment cache' }
                    else { $stack.Push($entry.FullName) }
                } elseif ($entry.Extension -in @('.pyc', '.pyo') -or $entry.Name -eq '.coverage' -or $entry.Name -match '^\.coverage\.') {
                    & $add $entry.FullName 'Python environment bytecode'
                }
            }
        }
    }

    foreach ($resourceRoot in @($ResourceRoots | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
        foreach ($profileRoot in @($resourceRoot, (Join-Path $resourceRoot 'studio'))) {
            foreach ($spec in @(
                @{ Directory = 'exports'; Pattern = '^\.export-'; Container = $true; Label = 'Stale checkpoint export staging' },
                @{ Directory = 'checkpoints'; Pattern = '^\.import-upload-'; Container = $false; Label = 'Stale checkpoint import upload' }
            )) {
                $directory = Join-Path $profileRoot $spec.Directory
                if (-not (Test-Path -LiteralPath $directory -PathType Container)) { continue }
                if (-not (Test-CleanupReparseChain -Path $directory -AllowedRoot $resourceRoot)) { & $add $directory $spec.Label; continue }
                try { $entries = @(Get-ChildItem -LiteralPath $directory -Force -ErrorAction Stop) } catch { & $add $directory $spec.Label -PlanError $_.Exception.Message; continue }
                foreach ($entry in $entries) {
                    if ($entry.Name -match $spec.Pattern -and (($entry.PSIsContainer -and $spec.Container) -or (-not $entry.PSIsContainer -and -not $spec.Container))) {
                        & $add $entry.FullName $spec.Label
                    }
                }
            }
        }
        $metadataRoot = Join-Path $resourceRoot 'models\huggingface\metadata'
        if (Test-Path -LiteralPath $metadataRoot -PathType Container) {
            if (-not (Test-CleanupReparseChain -Path $metadataRoot -AllowedRoot $resourceRoot)) {
                & $add $metadataRoot 'Model metadata temporary scan skipped'
            } else { try {
                foreach ($entry in Get-ChildItem -LiteralPath $metadataRoot -File -Force -ErrorAction Stop | Where-Object { $_.Name -match '\.[0-9a-fA-F]{32}\.tmp$' }) {
                    & $add $entry.FullName 'Stale atomic model metadata temporary'
                }
            } catch { & $add $metadataRoot 'Model metadata temporary scan failed' -PlanError $_.Exception.Message } }
        }
    }

    $bootstrapRoots = [Collections.Generic.List[string]]::new()
    [void]$bootstrapRoots.Add((Join-Path $RepoRoot 'settings'))
    if ($LocalAppData) { [void]$bootstrapRoots.Add((Join-Path $LocalAppData 'XREPORT\data')) }
    foreach ($bootstrapRoot in $bootstrapRoots) {
        if (-not (Test-Path -LiteralPath $bootstrapRoot -PathType Container)) { continue }
        if (-not (Test-CleanupReparseChain -Path $bootstrapRoot -AllowedRoot $RepoRoot)) { & $add $bootstrapRoot 'Launcher temporary-settings scan skipped'; continue }
        try { $temporarySettings = @(Get-ChildItem -LiteralPath $bootstrapRoot -File -Force -ErrorAction Stop | Where-Object { $_.Name -match '^\.(?:storage-[0-9a-fA-F]{32}|env\.[0-9a-fA-F]{32})\.tmp$' }) } catch { & $add $bootstrapRoot 'Launcher temporary-settings scan failed' -PlanError $_.Exception.Message; continue }
        foreach ($temporarySetting in $temporarySettings) { & $add $temporarySetting.FullName 'Stale launcher settings temporary' }
    }
    if ($LocalAppData) {
        $stateRoot = Join-Path (Join-Path $LocalAppData 'XREPORT\data') 'state'
        if (Test-Path -LiteralPath $stateRoot -PathType Container) {
            if (-not (Test-CleanupReparseChain -Path $stateRoot -AllowedRoot (Join-Path $LocalAppData 'XREPORT\data'))) {
                & $add $stateRoot 'Packaged desktop-state temporary scan skipped'
            } else { try {
                foreach ($entry in Get-ChildItem -LiteralPath $stateRoot -File -Force -ErrorAction Stop | Where-Object { $_.Name -match '^\.desktop-(?:ready|session)\.json\.[0-9a-fA-F]{16}\.tmp$' }) {
                    & $add $entry.FullName 'Stale packaged desktop state temporary'
                }
            } catch { & $add $stateRoot 'Packaged desktop-state temporary scan failed' -PlanError $_.Exception.Message } }
        }
    }

    $unique = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($target in $targets) { if (-not $unique.ContainsKey([string]$target.Path)) { $unique.Add([string]$target.Path, $target) } }
    return @($unique.Values)
}

function Get-XReportUninstallTargets {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [string]$LocalAppData,
        [string[]]$ResourceRoots = @(),
        [object[]]$AdditionalTargets = @(),
        [string[]]$PortableInstallPaths = @()
    )

    $root = [IO.Path]::GetFullPath($RepoRoot)
    $desktopRoot = Join-Path $root 'app\desktop'
    $desktopBuild = Join-Path $desktopRoot 'build'
    $tauriRoot = Join-Path $desktopRoot 'src-tauri'
    $targets = [Collections.Generic.List[object]]::new()
    [void]$targets.Add([pscustomobject]@{ Path = (Join-Path $root 'runtimes'); Label = 'Repository-managed portable runtimes and caches'; KeepRoot = $true; PreserveNames = @('.gitkeep') })
    foreach ($path in @(
        (Join-Path $root 'app\server\.venv'),
        (Join-Path $root 'app\server\.venv-studio'),
        (Join-Path $root '.venv'),
        (Join-Path $root 'app\client\node_modules'),
        (Join-Path $root 'app\client\dist'),
        (Join-Path $root 'app\client\dist-ssr'),
        (Join-Path $desktopRoot 'node_modules'),
        (Join-Path $desktopBuild '__pycache__'),
        (Join-Path $desktopBuild 'runtime-staging'),
        (Join-Path $desktopBuild 'pyinstaller'),
        (Join-Path $desktopBuild 'cpu-overlay'),
        (Join-Path $desktopBuild 'cargo-target'),
        (Join-Path $tauriRoot 'target'),
        (Join-Path $tauriRoot 'app'),
        (Join-Path $tauriRoot 'gen'),
        (Join-Path $tauriRoot 'ui'),
        (Join-Path $tauriRoot 'generated')
    )) {
        [void]$targets.Add([pscustomobject]@{ Path = $path; Label = 'Repository-generated runtime, dependency, or build output'; KeepRoot = $false; PreserveNames = @() })
    }

    $releaseDir = Join-Path $root 'release'
    $releasePattern = '^XREPORT-v\d+\.\d+\.\d+-windows-x64-(?:cpu|cuda)(?:-portable\.exe|\.msi|\.sha256|-build\.json)$'
    if (Test-Path -LiteralPath $releaseDir -PathType Container) {
        if (-not (Test-CleanupReparseChain -Path $releaseDir -AllowedRoot $root)) {
            [void]$targets.Add([pscustomobject]@{ Path = $releaseDir; Label = 'Desktop release scan skipped'; KeepRoot = $true; PreserveNames = @() })
        } else {
            try {
                foreach ($artifact in Get-ChildItem -LiteralPath $releaseDir -File -Force -ErrorAction Stop | Where-Object { $_.Name -match $releasePattern }) {
                    [void]$targets.Add([pscustomobject]@{ Path = $artifact.FullName; Label = 'Generated desktop release artifact'; KeepRoot = $false; PreserveNames = @() })
                }
            } catch {
                [void]$targets.Add([pscustomobject]@{ Path = $releaseDir; Label = 'Desktop release scan failed'; KeepRoot = $true; PreserveNames = @(); PlanError = $_.Exception.Message })
            }
        }
    }

    $dataRoot = if ([string]::IsNullOrWhiteSpace($LocalAppData)) { $null } else { Join-Path $LocalAppData 'XREPORT\data' }
    if ($dataRoot) {
        foreach ($path in @(
            (Join-Path $dataRoot 'runtimes\runtime'),
            (Join-Path $dataRoot 'state\desktop-ready.json'),
            (Join-Path $dataRoot 'state\desktop-session.json')
        )) {
            [void]$targets.Add([pscustomobject]@{ Path = $path; Label = 'Packaged-user runtime or transient state'; KeepRoot = $false; PreserveNames = @() })
        }
    }

    $coveredPaths = @($targets | ForEach-Object { [IO.Path]::GetFullPath([string]$_.Path) })
    foreach ($cacheTarget in @(Get-XReportCacheTargets -RepoRoot $root -LocalAppData $LocalAppData -ResourceRoots $ResourceRoots)) {
        $cachePath = [IO.Path]::GetFullPath([string]$cacheTarget.Path)
        $covered = $false
        foreach ($coveredPath in $coveredPaths) {
            $normalizedCovered = $coveredPath.TrimEnd('\', '/')
            if ($cachePath.Equals($normalizedCovered, [StringComparison]::OrdinalIgnoreCase) -or
                $cachePath.StartsWith($normalizedCovered + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
                $cachePath.StartsWith($normalizedCovered + [IO.Path]::AltDirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
                $covered = $true
                break
            }
        }
        if (-not $covered) { [void]$targets.Add($cacheTarget) }
    }

    foreach ($target in $AdditionalTargets) { [void]$targets.Add($target) }
    foreach ($portablePath in $PortableInstallPaths) {
        if ([string]::IsNullOrWhiteSpace($portablePath)) { continue }
        $fullPath = [IO.Path]::GetFullPath($portablePath)
        if ([IO.Path]::GetFileName($fullPath) -notmatch '(?i)^XREPORT-v\d+\.\d+\.\d+-windows-x64-(?:cpu|cuda)-portable\.exe$') {
            throw "PortableInstallPath must name a versioned XREPORT portable executable: $portablePath"
        }
        [void]$targets.Add([pscustomobject]@{ Path = $fullPath; Label = 'Explicit portable executable'; KeepRoot = $false; PreserveNames = @() })
    }

    $unique = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($target in $targets) {
        $key = [IO.Path]::GetFullPath([string]$target.Path)
        if (-not $unique.ContainsKey($key)) { $unique.Add($key, $target) }
    }
    return @($unique.Values)
}

function Select-XReportMsiProducts {
    [CmdletBinding()]
    param([object[]]$RegistryEntries = @())
    $products = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $RegistryEntries) {
        if ($null -eq $entry) { continue }
        $entryProperties = $entry.PSObject.Properties
        $displayName = if ($entryProperties['DisplayName']) { [string]$entryProperties['DisplayName'].Value } else { '' }
        $publisher = if ($entryProperties['Publisher']) { [string]$entryProperties['Publisher'].Value } else { '' }
        if ($displayName -notin @('XREPORT Inference CPU', 'XREPORT Inference CUDA') -or $publisher -ne 'CTCycle') { continue }
        $guidPattern = '\{[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\}'
        $uninstallString = if ($entryProperties['UninstallString']) { [string]$entryProperties['UninstallString'].Value } else { '' }
        $childName = if ($entryProperties['PSChildName']) { [string]$entryProperties['PSChildName'].Value } else { '' }
        $match = [regex]::Match($uninstallString, $guidPattern)
        $productCode = if ($match.Success) { $match.Value } elseif ($childName -match "^$guidPattern$") { $childName } else { '' }
        if (-not $productCode) { continue }
        if (-not $products.ContainsKey($productCode)) {
            $installLocation = if ($entryProperties['InstallLocation']) { [string]$entryProperties['InstallLocation'].Value } else { '' }
            $products.Add($productCode, [pscustomobject]@{ DisplayName = $displayName; Publisher = $publisher; ProductCode = $productCode; InstallLocation = $installLocation })
        }
    }
    return @($products.Values | Sort-Object DisplayName, ProductCode)
}

function Invoke-XReportMsiUninstall {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Product,
        [scriptblock]$StartProcess
    )
    if ($Product.Publisher -ne 'CTCycle' -or $Product.DisplayName -notin @('XREPORT Inference CPU', 'XREPORT Inference CUDA') -or $Product.ProductCode -notmatch '^\{[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\}$') {
        return [pscustomobject]@{ Product = $Product; Success = $false; RestartRequired = $false; ExitCode = $null; Message = 'Product did not pass the XREPORT MSI allowlist.' }
    }
    $arguments = @('/x', [string]$Product.ProductCode, '/passive', '/norestart')
    try {
        $process = if ($StartProcess) {
            & $StartProcess 'msiexec.exe' $arguments
        } else {
            Start-Process -FilePath 'msiexec.exe' -ArgumentList $arguments -Wait -PassThru -ErrorAction Stop
        }
        $exitCode = [int]$process.ExitCode
        return [pscustomobject]@{
            Product = $Product
            Success = $exitCode -in @(0, 3010)
            RestartRequired = $exitCode -eq 3010
            ExitCode = $exitCode
            Message = if ($exitCode -in @(0, 3010)) { '' } else { "msiexec exit $exitCode" }
        }
    } catch {
        return [pscustomobject]@{ Product = $Product; Success = $false; RestartRequired = $false; ExitCode = $null; Message = $_.Exception.Message }
    }
}

Export-ModuleMember -Function New-XReportCleanupPlan, Invoke-XReportCleanupPlan, Get-XReportApplicationDataTargets, Get-XReportCacheTargets, Get-XReportUninstallTargets, Resolve-XReportStoragePointer, Select-XReportMsiProducts, Invoke-XReportMsiUninstall
