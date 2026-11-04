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
    try {
        $cursor = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
        $root = [IO.Path]::GetFullPath($AllowedRoot).TrimEnd('\', '/')
    } catch { return $false }
    $withinAllowedRoot = Test-CleanupPathUnderRoot -Path $cursor -Root $root
    while (-not [string]::IsNullOrWhiteSpace($cursor)) {
        if (Test-Path -LiteralPath $cursor) {
            try {
                $item = Get-Item -LiteralPath $cursor -Force -ErrorAction Stop
                if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { return $false }
            } catch { return $false }
        }
        if ($withinAllowedRoot -and $cursor.Equals($root, [StringComparison]::OrdinalIgnoreCase)) { return $true }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) { break }
        $cursor = $parent
    }
    return (-not $withinAllowedRoot)
}

function Test-CleanupAncestorReparseChain {
    param([string]$Path, [string]$AllowedRoot)
    try {
        $cursor = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Path).TrimEnd('\', '/'))
        $root = [IO.Path]::GetFullPath($AllowedRoot).TrimEnd('\', '/')
    } catch { return $false }
    while (-not [string]::IsNullOrWhiteSpace($cursor)) {
        $cursor = $cursor.TrimEnd('\', '/')
        if (-not (Test-CleanupPathUnderRoot -Path $cursor -Root $root)) { return $false }
        try {
            if (([IO.File]::GetAttributes($cursor) -band [IO.FileAttributes]::ReparsePoint) -ne 0) { return $false }
        } catch [IO.FileNotFoundException] {
        } catch [IO.DirectoryNotFoundException] {
        } catch { return $false }
        if ($cursor.Equals($root, [StringComparison]::OrdinalIgnoreCase)) { return $true }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) { break }
        $cursor = $parent
    }
    return $false
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
            [void]$items.Add([pscustomobject]@{ Path = $pathValue; Label = $label; KeepRoot = $keepRoot; PreserveNames = $preserveNames; AllowedRoot = $null; Status = 'Failed'; Message = $_.Exception.Message })
            continue
        }
        $allowedRoot = $null
        foreach ($root in $roots) {
            if ((Test-CleanupPathUnderRoot -Path $target -Root $root) -and -not $target.TrimEnd('\', '/').Equals($root.TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase)) { $allowedRoot = $root; break }
        }
        if (-not $allowedRoot) {
            [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; KeepRoot = $keepRoot; PreserveNames = $preserveNames; AllowedRoot = $null; Status = 'Failed'; Message = 'Target is outside every allowlisted application root.' })
            continue
        }
        if (-not (Test-CleanupReparseChain -Path $target -AllowedRoot $allowedRoot)) {
            [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; KeepRoot = $keepRoot; PreserveNames = $preserveNames; AllowedRoot = $allowedRoot; Status = 'Skipped'; Message = 'A reparse point exists in the target path; it was not traversed.' })
            continue
        }
        if (-not [string]::IsNullOrWhiteSpace($planError)) {
            [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; KeepRoot = $keepRoot; PreserveNames = $preserveNames; AllowedRoot = $allowedRoot; Status = 'Failed'; Message = $planError })
            continue
        }
        try { $null = Get-Item -LiteralPath $target -Force -ErrorAction Stop }
        catch {
            if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound) {
                [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; KeepRoot = $keepRoot; PreserveNames = $preserveNames; AllowedRoot = $allowedRoot; Status = 'Skipped'; Message = 'Target does not exist.' })
            } else {
                [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; KeepRoot = $keepRoot; PreserveNames = $preserveNames; AllowedRoot = $allowedRoot; Status = 'Failed'; Message = $_.Exception.Message })
            }
            continue
        }
        [void]$items.Add([pscustomobject]@{ Path = $target; Label = $label; KeepRoot = $keepRoot; PreserveNames = $preserveNames; AllowedRoot = $allowedRoot; Status = 'Ready'; Message = '' })
    }

    return [pscustomobject]@{ Items = @($items); AllowedRoots = $roots; CreatedAt = [DateTime]::UtcNow }
}

function Add-CleanupFailure {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][System.Collections.Generic.List[object]]$Errors,
        [Parameter(Mandatory = $true)][pscustomobject]$TargetResult,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Message,
        [string]$Kind = 'Entry'
    )
    $TargetResult.FailedCount++
    $failure = [pscustomobject]@{ Path = $Path; Label = $TargetResult.Label; Kind = $Kind; Status = 'Failed'; Message = $Message }
    [void]$Errors.Add($failure)
    Write-Verbose "[Failed] $Path`: $Message"
}

function Remove-CleanupFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    $attributes = [IO.File]::GetAttributes($Path)
    if (($attributes -band [IO.FileAttributes]::ReadOnly) -ne 0) {
        [IO.File]::SetAttributes($Path, $attributes -band (-bnot [IO.FileAttributes]::ReadOnly))
    }
    [IO.File]::Delete($Path)
}

function Invoke-CleanupTargetWalk {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Target,
        [Parameter(Mandatory = $true)][pscustomobject]$TargetResult,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][System.Collections.Generic.List[object]]$Errors,
        [switch]$DryRun
    )

    if (-not (Test-CleanupReparseChain -Path $Target.Path -AllowedRoot $Target.AllowedRoot)) {
        $TargetResult.SkippedCount++
        $TargetResult.Message = 'A reparse point appeared in the target path after validation; it was not traversed.'
        Write-Verbose "[Skipped] $($Target.Path): $($TargetResult.Message)"
        return
    }

    $stack = [Collections.Generic.Stack[object]]::new()
    $stack.Push([pscustomobject]@{ Path = $Target.Path; IsRoot = $true; IsDirectory = $false; Initialized = $false; Enumerator = $null; Retained = $false; Parent = $null })
    while ($stack.Count -gt 0) {
        $frame = $stack.Peek()
        if (-not $frame.Initialized) {
            if (-not (Test-CleanupPathUnderRoot -Path $frame.Path -Root $Target.AllowedRoot)) {
                Add-CleanupFailure -Errors $Errors -TargetResult $TargetResult -Path $frame.Path -Message 'Entry escaped its validated cleanup root.'
                $frame.Retained = $true
                $frame.Initialized = $true
                $frame.IsDirectory = $false
            } else {
                try { $attributes = [IO.File]::GetAttributes($frame.Path) }
                catch {
                    Add-CleanupFailure -Errors $Errors -TargetResult $TargetResult -Path $frame.Path -Message $_.Exception.Message
                    $frame.Retained = $true
                    $frame.Initialized = $true
                    $frame.IsDirectory = $false
                    continue
                }

                if (($attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    $TargetResult.SkippedCount++
                    $frame.Retained = $true
                    Write-Verbose "[Skipped] $($frame.Path): Reparse point left untouched."
                    $frame.Initialized = $true
                    $frame.IsDirectory = $false
                } elseif (-not $frame.IsRoot -and $Target.PreserveNames -contains [IO.Path]::GetFileName($frame.Path)) {
                    $TargetResult.PreservedCount++
                    $frame.Retained = $true
                    Write-Verbose "[Preserved] $($frame.Path): Preserve-name rule."
                    $frame.Initialized = $true
                    $frame.IsDirectory = $false
                } elseif (($attributes -band [IO.FileAttributes]::Directory) -ne 0) {
                    if (-not (Test-CleanupAncestorReparseChain -Path $frame.Path -AllowedRoot $Target.AllowedRoot)) {
                        $TargetResult.SkippedCount++
                        $frame.Retained = $true
                        Write-Verbose "[Skipped] $($frame.Path): Reparse point found in its parent chain."
                        $frame.Initialized = $true
                        $frame.IsDirectory = $false
                    } else {
                        $frame.IsDirectory = $true
                        try {
                            $frame.Enumerator = ([IO.DirectoryInfo]::new($frame.Path).EnumerateFileSystemInfos()).GetEnumerator()
                            $frame.Initialized = $true
                        } catch {
                            Add-CleanupFailure -Errors $Errors -TargetResult $TargetResult -Path $frame.Path -Message "Cannot enumerate directory: $($_.Exception.Message)" -Kind 'Enumeration'
                            $frame.Retained = $true
                            $frame.Initialized = $true
                        }
                    }
                } else {
                    $TargetResult.PlannedCount++
                    if ($DryRun) {
                        Write-Verbose "[Planned] $($frame.Path)"
                    } else {
                        try {
                            $currentAttributes = [IO.File]::GetAttributes($frame.Path)
                            if (($currentAttributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                                $TargetResult.SkippedCount++
                                $frame.Retained = $true
                                Write-Verbose "[Skipped] $($frame.Path): Entry became a reparse point."
                            } else {
                                Remove-CleanupFile -Path $frame.Path
                                $TargetResult.CleanedCount++
                                Write-Verbose "[Cleaned] $($frame.Path)"
                            }
                        } catch {
                            Add-CleanupFailure -Errors $Errors -TargetResult $TargetResult -Path $frame.Path -Message $_.Exception.Message
                            $frame.Retained = $true
                        }
                    }
                    $frame.Initialized = $true
                    $frame.IsDirectory = $false
                }
            }

            if (-not $frame.IsDirectory) {
                $parent = $frame.Parent
                $retained = [bool]$frame.Retained
                [void]$stack.Pop()
                if ($retained -and $null -ne $parent) { $parent.Retained = $true }
                continue
            }
        }

        if ($frame.IsDirectory -and $null -ne $frame.Enumerator) {
            try { $hasNext = $frame.Enumerator.MoveNext() }
            catch {
                Add-CleanupFailure -Errors $Errors -TargetResult $TargetResult -Path $frame.Path -Message "Cannot enumerate directory: $($_.Exception.Message)" -Kind 'Enumeration'
                $frame.Retained = $true
                $hasNext = $false
            }
            if ($hasNext) {
                $entry = $frame.Enumerator.Current
                $stack.Push([pscustomobject]@{ Path = $entry.FullName; IsRoot = $false; IsDirectory = $false; Initialized = $false; Enumerator = $null; Retained = $false; Parent = $frame })
                continue
            }
            try { $frame.Enumerator.Dispose() } catch { }
            $frame.Enumerator = $null
        }

        if ($frame.IsDirectory) {
            if ($frame.IsRoot -and $Target.KeepRoot) {
                $retained = $false
            } elseif ($frame.Retained) {
                $retained = $true
            } else {
                $TargetResult.PlannedCount++
                if ($DryRun) {
                    $retained = $false
                    Write-Verbose "[Planned] $($frame.Path)"
                } else {
                    try {
                        $currentAttributes = [IO.File]::GetAttributes($frame.Path)
                        if (($currentAttributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                            $TargetResult.SkippedCount++
                            $retained = $true
                            Write-Verbose "[Skipped] $($frame.Path): Entry became a reparse point."
                        } else {
                            [IO.Directory]::Delete($frame.Path, $false)
                            $TargetResult.CleanedCount++
                            $retained = $false
                            Write-Verbose "[Cleaned] $($frame.Path)"
                        }
                    } catch {
                        Add-CleanupFailure -Errors $Errors -TargetResult $TargetResult -Path $frame.Path -Message "Directory changed or could not be removed: $($_.Exception.Message)"
                        $retained = $true
                    }
                }
            }
            $parent = $frame.Parent
            [void]$stack.Pop()
            if ($retained -and $null -ne $parent) { $parent.Retained = $true }
        }
    }
}

function Invoke-XReportCleanupPlan {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][pscustomobject]$Plan, [switch]$DryRun)

    $targetResults = [Collections.Generic.List[object]]::new()
    $errors = [Collections.Generic.List[object]]::new()
    foreach ($item in $Plan.Items) {
        $targetResult = [pscustomobject]@{
            Path = $item.Path; Label = $item.Label; Status = $item.Status; Message = $item.Message
            PlannedCount = 0; CleanedCount = 0; SkippedCount = 0; FailedCount = 0; PreservedCount = 0; MissingTargetCount = 0
        }
        [void]$targetResults.Add($targetResult)
        if ($item.Status -ne 'Ready') {
            if ($item.Status -eq 'Skipped') {
                $targetResult.SkippedCount++
                if ($item.Message -eq 'Target does not exist.') { $targetResult.MissingTargetCount++ }
                Write-Verbose "[Skipped] $($item.Path): $($item.Message)"
            } else {
                $targetResult.FailedCount++
                [void]$errors.Add([pscustomobject]@{ Path = $item.Path; Label = $item.Label; Kind = 'Target'; Status = 'Failed'; Message = $item.Message })
                Write-Verbose "[Failed] $($item.Path): $($item.Message)"
            }
            continue
        }
        Invoke-CleanupTargetWalk -Target $item -TargetResult $targetResult -Errors $errors -DryRun:$DryRun
    }
    $plannedCount = 0; $cleanedCount = 0; $skippedCount = 0; $failedCount = 0; $preservedCount = 0; $missingTargetCount = 0
    foreach ($targetResult in $targetResults) {
        $plannedCount += $targetResult.PlannedCount; $cleanedCount += $targetResult.CleanedCount
        $skippedCount += $targetResult.SkippedCount; $failedCount += $targetResult.FailedCount
        $preservedCount += $targetResult.PreservedCount; $missingTargetCount += $targetResult.MissingTargetCount
    }
    return [pscustomobject]@{
        DryRun = [bool]$DryRun
        TargetResults = @($targetResults)
        Errors = @($errors)
        PlannedCount = $plannedCount
        CleanedCount = $cleanedCount
        SkippedCount = $skippedCount
        FailedCount = $failedCount
        PreservedCount = $preservedCount
        MissingTargetCount = $missingTargetCount
    }
}

function Get-XReportCacheTargets {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [string]$LocalAppData,
        [string[]]$ResourceRoots = @(),
        [string[]]$ExcludedRoots = @()
    )
    $targets = [Collections.Generic.List[object]]::new()
    $excludedRoots = @($ExcludedRoots | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\', '/') } | Select-Object -Unique)
    $isExcluded = {
        param([string]$Path)
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        try { $fullPath = [IO.Path]::GetFullPath($Path) } catch { return $false }
        foreach ($excludedRoot in $excludedRoots) {
            $normalizedPath = $fullPath.TrimEnd('\', '/')
            if ($normalizedPath.Equals($excludedRoot, [StringComparison]::OrdinalIgnoreCase) -or
                $normalizedPath.StartsWith($excludedRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
                $normalizedPath.StartsWith($excludedRoot + [IO.Path]::AltDirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { return $true }
        }
        return $false
    }.GetNewClosure()
    $add = { param([string]$Path, [string]$Label, [switch]$KeepRoot, [string]$PlanError)
        if (-not [string]::IsNullOrWhiteSpace($Path) -and -not (& $isExcluded $Path)) { [void]$targets.Add([pscustomobject]@{ Path = $Path; Label = $Label; KeepRoot = [bool]$KeepRoot; PreserveNames = @('.gitkeep'); PlanError = $PlanError }) }
    }.GetNewClosure()
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
    if (-not (& $isExcluded $launcherRuntimeDir) -and (Test-Path -LiteralPath $launcherRuntimeDir -PathType Container)) {
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
        if (& $isExcluded $scanRoot) { continue }
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
        if (& $isExcluded $bytecodeRoot) { continue }
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
    foreach ($cacheTarget in @(Get-XReportCacheTargets -RepoRoot $root -LocalAppData $LocalAppData -ResourceRoots $ResourceRoots -ExcludedRoots $coveredPaths)) {
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
