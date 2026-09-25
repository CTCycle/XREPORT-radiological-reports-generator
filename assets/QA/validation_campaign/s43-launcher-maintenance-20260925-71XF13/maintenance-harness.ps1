$ErrorActionPreference = 'Stop'

$evidenceRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$repoRoot = [IO.Path]::GetFullPath((Join-Path $evidenceRoot '..\..\..\..')).TrimEnd('\')
$repoPrefix = $repoRoot + '\'
$fixtureRoot = [IO.Path]::GetFullPath((Join-Path $evidenceRoot 'harness-fixture'))
$fixturePrefix = $evidenceRoot + '\'
if (-not $fixtureRoot.StartsWith($fixturePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing fixture outside evidence directory: $fixtureRoot"
}
if (Test-Path -LiteralPath $fixtureRoot) {
    throw "Refusing to reuse an existing fixture directory: $fixtureRoot"
}

$launcherPath = Join-Path $repoRoot 'start_on_windows.ps1'
$tokens = $null
$parseErrors = $null
$launcherAst = [System.Management.Automation.Language.Parser]::ParseFile(
    $launcherPath,
    [ref]$tokens,
    [ref]$parseErrors
)
if ($parseErrors.Count -gt 0) {
    throw "Launcher parse failed: $($parseErrors[0].Message)"
}

$functionNames = @(
    'Clear-ApplicationCache',
    'Confirm-DestructiveAction',
    'Get-XReportApplicationProcessIds',
    'Get-XReportProcessTreeIds',
    'Invoke-RemoveDesktopRelease',
    'Remove-LauncherPath',
    'Stop-XReportProcesses',
    'Write-Info',
    'Write-Ok',
    'Write-Warn'
)
$functionNodes = @($launcherAst.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $functionNames -contains $node.Name
}, $true))
foreach ($name in $functionNames) {
    if (-not ($functionNodes | Where-Object Name -eq $name)) {
        throw "Launcher function not found: $name"
    }
}
foreach ($node in $functionNodes) {
    Invoke-Expression $node.Extent.Text
}

$global:OriginalRemoveLauncherPath = (Get-Command Remove-LauncherPath).ScriptBlock
$global:RemovalCalls = [Collections.Generic.List[object]]::new()
function Remove-LauncherPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$KeepRoot,
        [string[]]$PreserveNames = @('.gitkeep'),
        [switch]$Strict,
        [switch]$WhatIf,
        [string]$Activity = 'XREPORT: remove files'
    )
    [void]$global:RemovalCalls.Add([pscustomobject]@{
        Path = [string]$Path
        WhatIf = [bool]$WhatIf
    })
    Write-Host "Harness remove call: path='$Path' WhatIf=$WhatIf"
    & $global:OriginalRemoveLauncherPath @PSBoundParameters
}

$script:LauncherInteractive = $true
function Clear-LauncherProgress {}
function Read-Host { param([string]$Prompt); return 'y' }
function Start-LauncherProgress { param([string]$Activity, [string]$Status); return 1 }
function Update-LauncherProgress { param([int]$Id, [string]$Activity, [string]$Status, [int]$PercentComplete) }
function Complete-LauncherProgress { param([int]$Id) }
function Confirm-DestructiveAction { param([string]$Description); return $true }
function Get-LegacyCacheDirectories {
    Write-Host "Harness sees cache variable: local='$RuntimeCacheDir' global='$global:RuntimeCacheDir'"
    return @($global:LegacyCacheFixture | Where-Object { Test-Path -LiteralPath $_.FullName })
}
function Initialize-Environment {}

$global:ProcessFixture = @()
$global:KillRequests = [Collections.Generic.List[string]]::new()
$global:PortCalls = [Collections.Generic.List[int]]::new()
function Import-XReportEnvironment {
    return @{ FASTAPI_PORT = '59991'; UI_PORT = '59992' }
}
function Get-XReportProcessTable { return @($global:ProcessFixture) }
function Get-ProtectedLauncherProcessIds { param([object[]]$ProcessTable); return @() }
function Get-PortProcessId { param([int]$Port); return $null }
function Stop-PortListener { param([int]$Port, [int[]]$ExcludeProcessIds); [void]$global:PortCalls.Add($Port) }
function taskkill.exe { [void]$global:KillRequests.Add(($args -join ' ')); return 0 }

$global:RepoRoot = $repoRoot
$results = [ordered]@{
    source = $launcherPath
    fixture_root = $fixtureRoot
    cache = $null
    cache_multi_target = $null
    desktop_release = $null
    kill_processes = $null
}

try {
    New-Item -ItemType Directory -Path $fixtureRoot -Force | Out-Null

    # Exercise the exact cache action against a disposable replacement for its fixed root.
    $cacheRoot = Join-Path $fixtureRoot 'cache-case\runtime-cache'
    New-Item -ItemType Directory -Path (Join-Path $cacheRoot 'python'), (Join-Path $cacheRoot 'pip') -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $cacheRoot '.gitkeep') -Value '' -NoNewline
    Set-Content -LiteralPath (Join-Path $cacheRoot 'root.tmp') -Value 'temporary' -NoNewline
    Set-Content -LiteralPath (Join-Path $cacheRoot 'python\state.bin') -Value 'temporary' -NoNewline
    Set-Content -LiteralPath (Join-Path $cacheRoot 'pip\.gitkeep') -Value '' -NoNewline
    Set-Content -LiteralPath (Join-Path $cacheRoot 'pip\temporary.whl') -Value 'temporary' -NoNewline
    $global:RuntimeCacheDir = $cacheRoot
    $RuntimeCacheDir = $cacheRoot
    $global:LegacyCacheFixture = @()
    $cacheFullPath = [IO.Path]::GetFullPath($global:RuntimeCacheDir)
    $cacheProbe = Remove-LauncherPath -Path $cacheRoot -PreserveNames @('.gitkeep') -WhatIf
    Write-Host "Helper planned $($cacheProbe.Planned) fixture entries."
    if (-not $cacheFullPath.StartsWith($fixturePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cache action escaped the disposable fixture: $cacheFullPath"
    }
    $global:RemovalCalls.Clear()
    Clear-ApplicationCache
    $cacheActionCall = if ($global:RemovalCalls.Count -gt 0) { [string]$global:RemovalCalls[0].Path } else { $null }
    $cacheRemaining = @(Get-ChildItem -LiteralPath $cacheRoot -Force -Recurse |
        ForEach-Object { $_.FullName.Substring($cacheRoot.Length + 1).Replace('\', '/') } |
        Sort-Object)
    $expectedCacheRemaining = @('pip', 'pip/.gitkeep', '.gitkeep') | Sort-Object
    $cachePass = ($cacheActionCall -eq $cacheFullPath) -and
        (($cacheRemaining -join '|') -eq ($expectedCacheRemaining -join '|'))
    $cacheNoOp = $false
    if ($cachePass) {
        $global:RemovalCalls.Clear()
        Clear-ApplicationCache
        $cacheAfterNoOp = @(Get-ChildItem -LiteralPath $cacheRoot -Force -Recurse |
            ForEach-Object { $_.FullName.Substring($cacheRoot.Length + 1).Replace('\', '/') } |
            Sort-Object)
        $cacheNoOp = (($cacheAfterNoOp -join '|') -eq ($expectedCacheRemaining -join '|'))
    }
    $results.cache = @{
        cleared = $cachePass
        target_count = 1
        preserved_gitkeep = ($cacheRemaining -contains '.gitkeep') -and
            ($cacheRemaining -contains 'pip/.gitkeep')
        idempotent_no_op = $cacheNoOp
        observed_target = $cacheActionCall
        remaining_after_first_run = $cacheRemaining
        legacy_roots_touched = $false
    }

    # Exercise the same action with its runtime cache and two existing legacy roots.
    $multiCacheRoot = Join-Path $fixtureRoot 'cache-multi\runtime-cache'
    $multiLegacyRoots = @(
        (Join-Path $fixtureRoot 'cache-multi\legacy-old-a'),
        (Join-Path $fixtureRoot 'cache-multi\legacy-old-b')
    )
    $multiTargets = @($multiCacheRoot) + $multiLegacyRoots
    foreach ($target in $multiTargets) {
        if (-not [IO.Path]::GetFullPath($target).StartsWith($fixturePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Multi-target cache fixture escaped the evidence directory: $target"
        }
    }
    $multiCacheDirectories = @((Join-Path $multiCacheRoot 'python')) + $multiLegacyRoots
    New-Item -ItemType Directory -Path $multiCacheDirectories -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $multiCacheRoot '.gitkeep') -Value '' -NoNewline
    Set-Content -LiteralPath (Join-Path $multiCacheRoot 'root.tmp') -Value 'temporary' -NoNewline
    Set-Content -LiteralPath (Join-Path $multiCacheRoot 'python\state.bin') -Value 'temporary' -NoNewline
    Set-Content -LiteralPath (Join-Path $multiLegacyRoots[0] 'old-a.tmp') -Value 'temporary' -NoNewline
    Set-Content -LiteralPath (Join-Path $multiLegacyRoots[1] 'old-b.tmp') -Value 'temporary' -NoNewline
    $global:RuntimeCacheDir = $multiCacheRoot
    $RuntimeCacheDir = $multiCacheRoot
    $global:LegacyCacheFixture = @($multiLegacyRoots | ForEach-Object { [pscustomobject]@{ FullName = $_ } })
    $expectedMultiTargets = @($multiTargets | ForEach-Object { [IO.Path]::GetFullPath($_) })
    $global:RemovalCalls.Clear()
    Clear-ApplicationCache
    $multiObservedTargets = @($global:RemovalCalls | ForEach-Object { [IO.Path]::GetFullPath($_.Path) })
    $multiCacheRemaining = @(Get-ChildItem -LiteralPath $multiCacheRoot -Force -Recurse |
        ForEach-Object { $_.FullName.Substring($multiCacheRoot.Length + 1).Replace('\', '/') } |
        Sort-Object)
    $legacyRootsRemain = @($multiLegacyRoots | Where-Object { Test-Path -LiteralPath $_ })
    $multiPass = (($multiObservedTargets -join '|') -eq ($expectedMultiTargets -join '|')) -and
        (($multiCacheRemaining -join '|') -eq '.gitkeep') -and
        ($legacyRootsRemain.Count -eq 0)
    $global:RemovalCalls.Clear()
    Clear-ApplicationCache
    $multiNoOpTargets = @($global:RemovalCalls | ForEach-Object { [IO.Path]::GetFullPath($_.Path) })
    $multiNoOp = ($multiNoOpTargets.Count -eq 1) -and
        ($multiNoOpTargets[0] -eq [IO.Path]::GetFullPath($multiCacheRoot)) -and
        ((@(Get-ChildItem -LiteralPath $multiCacheRoot -Force -Recurse |
            ForEach-Object { $_.FullName.Substring($multiCacheRoot.Length + 1).Replace('\', '/') } |
            Sort-Object) -join '|') -eq '.gitkeep')
    $results.cache_multi_target = @{
        cleared = $multiPass
        target_count = $multiObservedTargets.Count
        observed_targets = $multiObservedTargets
        remaining_runtime_cache = $multiCacheRemaining
        legacy_roots_removed = ($legacyRootsRemain.Count -eq 0)
        idempotent_no_op = $multiNoOp
        repeat_targets = $multiNoOpTargets
    }

    # Exercise the exact release-removal action with every target redirected into the fixture.
    $desktopRoot = Join-Path $fixtureRoot 'desktop-case'
    $global:DesktopReleaseDir = Join-Path $desktopRoot 'release'
    $global:DesktopBuildDir = Join-Path $desktopRoot 'build'
    $global:DesktopTauriDir = Join-Path $desktopRoot 'src-tauri'
    $global:DesktopTargetDir = Join-Path $global:DesktopTauriDir 'target'
    $releaseTargets = @(
        $global:DesktopReleaseDir,
        (Join-Path $global:DesktopBuildDir 'runtime-staging'),
        (Join-Path $global:DesktopBuildDir 'pyinstaller'),
        (Join-Path $global:DesktopBuildDir 'cpu-overlay'),
        (Join-Path $global:DesktopBuildDir 'cargo-target'),
        $global:DesktopTargetDir,
        (Join-Path $global:DesktopTauriDir 'generated\runtime.zip'),
        (Join-Path $global:DesktopTauriDir 'ui')
    )
    foreach ($target in $releaseTargets) {
        if (-not [IO.Path]::GetFullPath($target).StartsWith($fixturePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Release cleanup target escaped the disposable fixture: $target"
        }
    }
    New-Item -ItemType Directory -Path (
        Join-Path $global:DesktopReleaseDir 'payload'
    ), (
        Join-Path $global:DesktopBuildDir 'runtime-staging\cpu'
    ), (
        Join-Path $global:DesktopBuildDir 'pyinstaller\cpu'
    ), (
        Join-Path $global:DesktopBuildDir 'cpu-overlay'
    ), (
        Join-Path $global:DesktopBuildDir 'cargo-target\cpu\release'
    ), (
        Join-Path $global:DesktopTargetDir 'debug'
    ), (
        Join-Path $global:DesktopTauriDir 'generated'
    ), (
        Join-Path $global:DesktopTauriDir 'ui'
    ) -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $global:DesktopReleaseDir 'payload\sample.exe') -Value 'fixture' -NoNewline
    Set-Content -LiteralPath (Join-Path $global:DesktopBuildDir 'runtime-staging\cpu\sample.bin') -Value 'fixture' -NoNewline
    Set-Content -LiteralPath (Join-Path $global:DesktopBuildDir 'pyinstaller\cpu\sample.bin') -Value 'fixture' -NoNewline
    Set-Content -LiteralPath (Join-Path $global:DesktopBuildDir 'cpu-overlay\sample.bin') -Value 'fixture' -NoNewline
    Set-Content -LiteralPath (Join-Path $global:DesktopBuildDir 'cargo-target\cpu\release\sample.exe') -Value 'fixture' -NoNewline
    Set-Content -LiteralPath (Join-Path $global:DesktopTargetDir 'debug\sample.exe') -Value 'fixture' -NoNewline
    Set-Content -LiteralPath (Join-Path $global:DesktopTauriDir 'generated\runtime.zip') -Value 'fixture' -NoNewline
    Set-Content -LiteralPath (Join-Path $global:DesktopTauriDir 'ui\index.html') -Value 'fixture' -NoNewline
    Set-Content -LiteralPath (Join-Path $global:DesktopBuildDir 'tauri-cpu-3.1.0.json') -Value '{}' -NoNewline
    Set-Content -LiteralPath (Join-Path $desktopRoot 'keep.txt') -Value 'outside cleanup targets' -NoNewline
    Invoke-RemoveDesktopRelease
    $releaseTargetsRemain = @($releaseTargets | Where-Object { Test-Path -LiteralPath $_ })
    if ($releaseTargetsRemain.Count -gt 0) {
        throw "RemoveDesktopRelease left target(s): $($releaseTargetsRemain -join ', ')"
    }
    if (Test-Path -LiteralPath (Join-Path $global:DesktopBuildDir 'tauri-cpu-3.1.0.json')) {
        throw 'RemoveDesktopRelease left the generated Tauri config.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $desktopRoot 'keep.txt'))) {
        throw 'RemoveDesktopRelease changed a sibling outside its target list.'
    }
    Invoke-RemoveDesktopRelease
    $results.desktop_release = @{
        removed_all_fixture_targets = $true
        removed_generated_config = $true
        preserved_sibling = $true
        idempotent_no_op = $true
    }

    # Intercept every process/port termination boundary; synthetic PIDs are guaranteed absent.
    $backendPid = 2000000000
    $childPid = 2000000001
    if ((Get-Process -Id $backendPid -ErrorAction SilentlyContinue) -or
        (Get-Process -Id $childPid -ErrorAction SilentlyContinue)) {
        throw 'Synthetic process ID unexpectedly exists; refusing process-action harness.'
    }
    $appPath = Join-Path $repoRoot 'app'
    $pythonPath = Join-Path $repoRoot 'app\server\.venv\Scripts\python.exe'
    $global:ProcessFixture = @(
        [pscustomobject]@{
            ProcessId = $backendPid
            ParentProcessId = 1
            Name = 'python.exe'
            ExecutablePath = $pythonPath
            CommandLine = "$pythonPath -m uvicorn server.app:app --app-dir $appPath"
        },
        [pscustomobject]@{
            ProcessId = $childPid
            ParentProcessId = $backendPid
            Name = 'python.exe'
            ExecutablePath = 'C:\Python\python.exe'
            CommandLine = 'python.exe worker-child'
        },
        [pscustomobject]@{
            ProcessId = 2000000002
            ParentProcessId = 1
            Name = 'python.exe'
            ExecutablePath = 'C:\Python\python.exe'
            CommandLine = 'python.exe -m uvicorn unrelated.service:app'
        },
        [pscustomobject]@{
            ProcessId = 2000000003
            ParentProcessId = 1
            Name = 'node.exe'
            ExecutablePath = (Join-Path $repoRoot 'runtimes\nodejs\node.exe')
            CommandLine = 'node.exe npm install'
        }
    )
    $selected = @(Get-XReportApplicationProcessIds -ProcessTable $global:ProcessFixture)
    if (($selected -join ',') -ne [string]$backendPid) {
        throw "KillProcesses selected unexpected roots: $($selected -join ',')"
    }
    $tree = @(Get-XReportProcessTreeIds -ProcessTable $global:ProcessFixture -RootProcessIds $selected)
    if (($tree -join ',') -ne "$backendPid,$childPid") {
        throw "KillProcesses synthetic tree mismatch: $($tree -join ',')"
    }
    Stop-XReportProcesses
    if ($global:KillRequests.Count -ne 1 -or $global:KillRequests[0] -ne "/PID $backendPid /T /F") {
        throw "KillProcesses termination boundary mismatch: $($global:KillRequests -join ';')"
    }
    $positiveKillRequest = $global:KillRequests[0]

    $global:ProcessFixture = @(
        [pscustomobject]@{
            ProcessId = 2000000004
            ParentProcessId = 1
            Name = 'pwsh.exe'
            ExecutablePath = 'C:\PowerShell\pwsh.exe'
            CommandLine = 'pwsh.exe -NoProfile -Command idle'
        }
    )
    $global:KillRequests.Clear()
    $global:PortCalls.Clear()
    Stop-XReportProcesses
    if ($global:KillRequests.Count -ne 0) {
        throw 'KillProcesses no-op sent a termination request.'
    }
    $results.kill_processes = @{
        selected_only_repo_backend = $selected
        synthetic_tree = $tree
        intercepted_kill = $positiveKillRequest
        no_op_made_no_kill_request = $true
        port_termination_intercepted = $true
    }
}
finally {
    if (Test-Path -LiteralPath $fixtureRoot) {
        $resolvedFixture = [IO.Path]::GetFullPath($fixtureRoot)
        if (-not $resolvedFixture.StartsWith($fixturePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing cleanup outside the evidence directory: $resolvedFixture"
        }
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
    }
}

$results.timestamp_utc = [DateTime]::UtcNow.ToString('o')
$results.fixture_removed = -not (Test-Path -LiteralPath $fixtureRoot)
$resultPath = Join-Path $evidenceRoot 'maintenance-harness-results.json'
$results | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $resultPath -Encoding utf8
$results | ConvertTo-Json -Depth 7
