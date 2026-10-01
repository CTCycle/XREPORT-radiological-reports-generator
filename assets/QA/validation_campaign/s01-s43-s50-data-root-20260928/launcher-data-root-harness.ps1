$ErrorActionPreference = 'Stop'

function Assert-Condition {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function New-FixtureFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$Content = 'fixture'
    )

    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Set-Content -LiteralPath $Path -Value $Content -NoNewline
}

$evidenceRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$fixturePrefix = $evidenceRoot + '\'
$fixtureRoot = [IO.Path]::GetFullPath((Join-Path $evidenceRoot 'harness-fixture'))
Assert-Condition $fixtureRoot.StartsWith($fixturePrefix, [StringComparison]::OrdinalIgnoreCase) 'Fixture escaped the evidence directory.'
Assert-Condition (-not (Test-Path -LiteralPath $fixtureRoot)) "Refusing to reuse fixture path: $fixtureRoot"

$launcherPath = [IO.Path]::GetFullPath((Join-Path $evidenceRoot '..\..\..\..\start_on_windows.ps1'))
$originalProcessRoot = $env:XREPORT_RESOURCES_DIR
$results = [ordered]@{
    status = 'FAIL'
    launcher = $launcherPath
    fixture_root = $fixtureRoot
}
$failure = $null

try {
    $script:RepoRoot = Join-Path $fixtureRoot 'repo'
    $global:RepoRoot = $script:RepoRoot
    $script:RuntimesDir = Join-Path $script:RepoRoot 'runtimes'
    $script:RuntimeCacheDir = Join-Path $script:RuntimesDir 'cache'
    $script:VenvDir = Join-Path $script:RepoRoot 'app\server\.venv'
    $script:ServerDir = Join-Path $script:RepoRoot 'app\server'
    $script:ClientDir = Join-Path $script:RepoRoot 'app\client'
    $script:DesktopDir = Join-Path $script:RepoRoot 'app\desktop'
    $script:DesktopTargetDir = Join-Path $script:DesktopDir 'target'

    New-Item -ItemType Directory -Path $script:RepoRoot -Force | Out-Null

    $tokens = $null
    $parseErrors = $null
    $launcherAst = [System.Management.Automation.Language.Parser]::ParseFile(
        $launcherPath,
        [ref]$tokens,
        [ref]$parseErrors
    )
    Assert-Condition ($parseErrors.Count -eq 0) "Launcher parse failed: $($parseErrors[0].Message)"

    $functionNames = @(
        'Clear-ApplicationCache',
        'Get-ConfiguredResourceRoot',
        'Get-LegacyCacheDirectories',
        'Get-XReportUserDataTargets',
        'Remove-AllData',
        'Remove-Checkpoints',
        'Remove-LauncherPath',
        'Remove-XReportUserDataTargets'
    )
    $functionNodes = @($launcherAst.FindAll({
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $functionNames -contains $node.Name
    }, $true))
    foreach ($name in $functionNames) {
        $functionFound = [bool]($functionNodes | Where-Object Name -eq $name)
        Assert-Condition -Condition $functionFound -Message "Launcher function not found: $name"
    }
    foreach ($node in $functionNodes) {
        Invoke-Expression $node.Extent.Text
    }

    $global:SettingsResourceValue = $null
    function Import-XReportEnvironment {
        $settings = @{}
        if (-not [string]::IsNullOrWhiteSpace([string]$global:SettingsResourceValue)) {
            $settings['XREPORT_RESOURCES_DIR'] = [string]$global:SettingsResourceValue
        }
        return $settings
    }
    function Confirm-DestructiveAction { return $true }
    function Start-LauncherProgress { return 1 }
    function Update-LauncherProgress {}
    function Complete-LauncherProgress {}
    function Initialize-Environment {}
    function Write-Info {}
    function Write-Ok {}
    function Write-Warn {}

    $global:RemovalCalls = [Collections.Generic.List[object]]::new()
    $global:OriginalRemoveLauncherPath = (Get-Command Remove-LauncherPath).ScriptBlock
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
            Path = [IO.Path]::GetFullPath($Path)
            KeepRoot = [bool]$KeepRoot
            WhatIf = [bool]$WhatIf
        })
        & $global:OriginalRemoveLauncherPath @PSBoundParameters
    }

    $env:XREPORT_RESOURCES_DIR = $null
    $global:SettingsResourceValue = $null
    $defaultRoot = Get-ConfiguredResourceRoot
    $expectedDefaultRoot = [IO.Path]::GetFullPath((Join-Path $script:RepoRoot 'data'))
    Assert-Condition ($defaultRoot -eq $expectedDefaultRoot) "Default root was '$defaultRoot'; expected '$expectedDefaultRoot'."

    $global:SettingsResourceValue = 'settings-data'
    $settingsRoot = Get-ConfiguredResourceRoot
    $expectedSettingsRoot = [IO.Path]::GetFullPath((Join-Path $script:RepoRoot 'settings-data'))
    Assert-Condition ($settingsRoot -eq $expectedSettingsRoot) 'The dotenv resource root did not resolve relative to the repository.'

    $overrideRoot = [IO.Path]::GetFullPath((Join-Path $fixtureRoot 'process-override'))
    $env:XREPORT_RESOURCES_DIR = $overrideRoot
    $processRoot = Get-ConfiguredResourceRoot
    Assert-Condition ($processRoot -eq $overrideRoot) 'The process-level resource root did not take precedence over dotenv.'
    $results.root_resolution = @{
        default = $defaultRoot
        dotenv = $settingsRoot
        process_override = $processRoot
        process_override_precedence = $processRoot -eq $overrideRoot
    }

    $defaultCache = Join-Path $defaultRoot 'cache'
    $defaultModelCache = Join-Path $defaultRoot 'models\huggingface\cache'
    $overrideCache = Join-Path $overrideRoot 'caches'
    New-FixtureFile (Join-Path $script:RuntimeCacheDir '.gitkeep') ''
    New-FixtureFile (Join-Path $script:RuntimeCacheDir 'temporary.bin')
    New-FixtureFile (Join-Path $defaultCache 'legacy.bin')
    New-FixtureFile (Join-Path $defaultModelCache 'download.bin')
    New-FixtureFile (Join-Path $overrideCache 'override.bin')
    New-FixtureFile (Join-Path $defaultRoot 'models\pinned\weights.bin') 'preserve-default-model'

    $global:RemovalCalls.Clear()
    Clear-ApplicationCache
    $cacheCalls = @($global:RemovalCalls | ForEach-Object { $_.Path })
    $expectedCacheTargets = @(
        [IO.Path]::GetFullPath($script:RuntimeCacheDir),
        [IO.Path]::GetFullPath($defaultCache),
        [IO.Path]::GetFullPath($defaultModelCache),
        [IO.Path]::GetFullPath($overrideCache)
    )
    foreach ($target in $cacheCalls) {
        Assert-Condition $target.StartsWith($fixturePrefix, [StringComparison]::OrdinalIgnoreCase) "Cache cleanup escaped the fixture: $target"
    }
    foreach ($target in $expectedCacheTargets) {
        Assert-Condition ($cacheCalls -contains $target) "Cache cleanup omitted expected target: $target"
    }
    Assert-Condition (-not (Test-Path -LiteralPath $defaultCache)) 'The relocated default data cache remained after cleanup.'
    Assert-Condition (-not (Test-Path -LiteralPath $defaultModelCache)) 'The relocated model cache remained after cleanup.'
    Assert-Condition (-not (Test-Path -LiteralPath $overrideCache)) 'The process-override cache remained after cleanup.'
    Assert-Condition (Test-Path -LiteralPath (Join-Path $script:RuntimeCacheDir '.gitkeep')) 'The runtime cache marker was not preserved.'
    Assert-Condition (Test-Path -LiteralPath (Join-Path $defaultRoot 'models\pinned\weights.bin')) 'Cache cleanup removed a non-cache model file.'

    $global:RemovalCalls.Clear()
    Clear-ApplicationCache
    $cacheRepeatCalls = @($global:RemovalCalls | ForEach-Object { $_.Path })
    Assert-Condition ($cacheRepeatCalls.Count -eq 1 -and $cacheRepeatCalls[0] -eq [IO.Path]::GetFullPath($script:RuntimeCacheDir)) 'Repeated cache cleanup was not an idempotent no-op.'
    $results.cache_cleanup = @{
        first_targets = $cacheCalls
        repeat_targets = $cacheRepeatCalls
        default_data_cache_removed = -not (Test-Path -LiteralPath $defaultCache)
        override_cache_removed = -not (Test-Path -LiteralPath $overrideCache)
        runtime_marker_preserved = Test-Path -LiteralPath (Join-Path $script:RuntimeCacheDir '.gitkeep')
        non_cache_model_preserved = Test-Path -LiteralPath (Join-Path $defaultRoot 'models\pinned\weights.bin')
        repeat_was_no_op = $cacheRepeatCalls.Count -eq 1
    }

    $checkpointsRoot = Join-Path $overrideRoot 'checkpoints'
    New-FixtureFile (Join-Path $checkpointsRoot '.gitkeep') ''
    New-FixtureFile (Join-Path $checkpointsRoot 'checkpoint.bin')
    New-FixtureFile (Join-Path $overrideRoot 'models\pinned\weights.bin') 'preserve-checkpoint-adjacent-model'
    $global:RemovalCalls.Clear()
    Remove-Checkpoints
    $checkpointCalls = @($global:RemovalCalls | ForEach-Object { $_.Path })
    Assert-Condition ($checkpointCalls.Count -eq 1 -and $checkpointCalls[0] -eq [IO.Path]::GetFullPath($checkpointsRoot)) 'Checkpoint cleanup selected an unexpected target.'
    Assert-Condition (-not (Test-Path -LiteralPath (Join-Path $checkpointsRoot 'checkpoint.bin'))) 'Checkpoint cleanup left its fixture file.'
    Assert-Condition (Test-Path -LiteralPath (Join-Path $checkpointsRoot '.gitkeep')) 'Checkpoint cleanup removed the directory marker.'
    Assert-Condition (Test-Path -LiteralPath (Join-Path $overrideRoot 'models\pinned\weights.bin')) 'Checkpoint cleanup touched model data.'
    $global:RemovalCalls.Clear()
    Remove-Checkpoints
    $checkpointRepeatCalls = @($global:RemovalCalls | ForEach-Object { $_.Path })
    Assert-Condition ($checkpointRepeatCalls.Count -eq 1 -and $checkpointRepeatCalls[0] -eq [IO.Path]::GetFullPath($checkpointsRoot)) 'Repeated checkpoint cleanup selected an unexpected target.'
    $results.checkpoint_cleanup = @{
        first_targets = $checkpointCalls
        repeat_targets = $checkpointRepeatCalls
        checkpoint_removed = -not (Test-Path -LiteralPath (Join-Path $checkpointsRoot 'checkpoint.bin'))
        model_preserved = Test-Path -LiteralPath (Join-Path $overrideRoot 'models\pinned\weights.bin')
    }

    New-FixtureFile (Join-Path $overrideRoot 'database.db')
    New-FixtureFile (Join-Path $overrideRoot 'database.db-wal')
    New-FixtureFile (Join-Path $overrideRoot 'database.db-shm')
    New-FixtureFile (Join-Path $overrideRoot 'database.db-journal')
    New-FixtureFile (Join-Path $checkpointsRoot 'checkpoint-after-reset.bin')
    New-FixtureFile (Join-Path $overrideRoot 'models\pinned\weights-after-reset.bin')
    New-FixtureFile (Join-Path $overrideRoot 'models\tokenizers\tokenizer.bin')
    New-FixtureFile (Join-Path $overrideRoot 'logs\application.log')
    New-FixtureFile (Join-Path $overrideRoot 'templates\preserved-template.bin')

    $global:RemovalCalls.Clear()
    Remove-AllData
    $dataCalls = @($global:RemovalCalls | ForEach-Object { $_.Path })
    foreach ($target in $dataCalls) {
        Assert-Condition $target.StartsWith([IO.Path]::GetFullPath($overrideRoot).TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) "Data cleanup escaped its process override: $target"
    }
    foreach ($target in @('database.db', 'database.db-wal', 'database.db-shm', 'database.db-journal')) {
        Assert-Condition (-not (Test-Path -LiteralPath (Join-Path $overrideRoot $target))) "Data cleanup left $target."
    }
    Assert-Condition (-not (Test-Path -LiteralPath (Join-Path $checkpointsRoot 'checkpoint-after-reset.bin'))) 'Data cleanup left a checkpoint.'
    Assert-Condition (-not (Test-Path -LiteralPath (Join-Path $overrideRoot 'models\pinned\weights-after-reset.bin'))) 'Data cleanup left a model.'
    Assert-Condition (-not (Test-Path -LiteralPath (Join-Path $overrideRoot 'models\tokenizers\tokenizer.bin'))) 'Data cleanup left a tokenizer.'
    Assert-Condition (-not (Test-Path -LiteralPath (Join-Path $overrideRoot 'logs\application.log'))) 'Data cleanup left a log.'
    Assert-Condition (Test-Path -LiteralPath (Join-Path $overrideRoot 'templates\preserved-template.bin')) 'Data cleanup removed an application template.'
    $global:RemovalCalls.Clear()
    Remove-AllData
    $dataRepeatCalls = @($global:RemovalCalls | ForEach-Object { $_.Path })
    $results.all_data_cleanup = @{
        first_targets = $dataCalls
        repeat_targets = $dataRepeatCalls
        disposable_database_removed = -not (Test-Path -LiteralPath (Join-Path $overrideRoot 'database.db'))
        checkpoints_removed = -not (Test-Path -LiteralPath (Join-Path $checkpointsRoot 'checkpoint-after-reset.bin'))
        model_and_log_data_removed = (-not (Test-Path -LiteralPath (Join-Path $overrideRoot 'models\pinned\weights-after-reset.bin'))) -and (-not (Test-Path -LiteralPath (Join-Path $overrideRoot 'logs\application.log')))
        template_preserved = Test-Path -LiteralPath (Join-Path $overrideRoot 'templates\preserved-template.bin')
        repeated_without_error = $true
    }

    $results.status = 'PASS'
}
catch {
    $failure = $_.Exception.Message
    $results.failure = $failure
}
finally {
    $env:XREPORT_RESOURCES_DIR = $originalProcessRoot
    if (Test-Path -LiteralPath $fixtureRoot) {
        $resolvedFixture = [IO.Path]::GetFullPath($fixtureRoot).TrimEnd('\')
        Assert-Condition $resolvedFixture.StartsWith($fixturePrefix, [StringComparison]::OrdinalIgnoreCase) "Refusing to remove fixture outside evidence directory: $resolvedFixture"
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
    }
    $results.fixture_removed = -not (Test-Path -LiteralPath $fixtureRoot)
    $receiptPath = Join-Path $evidenceRoot 'launcher-data-root-harness-receipt.json'
    [IO.File]::WriteAllText($receiptPath, ($results | ConvertTo-Json -Depth 8))
}

Get-Content -LiteralPath (Join-Path $evidenceRoot 'launcher-data-root-harness-receipt.json')
if ($failure) {
    throw $failure
}
