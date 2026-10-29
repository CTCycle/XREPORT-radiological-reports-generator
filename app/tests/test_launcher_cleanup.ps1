$ErrorActionPreference = 'Stop'
$modulePath = Join-Path $PSScriptRoot '..\scripts\launcher_cleanup.psm1'
Import-Module -Name $modulePath -Force

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Write-FixtureFile {
    param([string]$Path, [string]$Content = 'fixture')
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    [IO.File]::WriteAllText($Path, $Content)
}

$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) "xreport-cleanup-tests-$([guid]::NewGuid().ToString('N'))"
$repoRoot = Join-Path $fixtureRoot 'repository'
$localAppData = Join-Path $fixtureRoot 'profile\AppData\Local'
$appDataRoot = Join-Path $localAppData 'XREPORT\data'
$resourceRoot = Join-Path $fixtureRoot 'external-resources'
$junctionPath = Join-Path $repoRoot 'app\server\linked-cache'
$outsideRoot = Join-Path $fixtureRoot 'outside'
$lockedStream = $null
$aclSnapshot = $null
$aclTarget = Join-Path $fixtureRoot 'acl-denied'

try {
    foreach ($path in @($repoRoot, $appDataRoot, $resourceRoot, $outsideRoot)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
    $expectedCaches = @(
        (Join-Path $repoRoot 'runtimes\cache\huggingface\hub\snapshot\model.bin'),
        (Join-Path $repoRoot 'app\server\runtimes\cache\legacy.bin'),
        (Join-Path $repoRoot 'app\server\xreport\__pycache__\module.pyc'),
        (Join-Path $repoRoot 'app\server\.pytest_cache\v\cache\lastfailed'),
        (Join-Path $repoRoot 'app\server\.venv\Lib\site-packages\tool\.mypy_cache\cache.db'),
        (Join-Path $repoRoot 'app\server\assets\QA\pytest-cache-fixture\cache.json'),
        (Join-Path $repoRoot 'app\client\.angular\cache\angular.bin'),
        (Join-Path $repoRoot 'app\client\node_modules\.cache\bundler.bin'),
        (Join-Path $repoRoot 'app\client\node_modules\.vite\deps\_metadata.json'),
        (Join-Path $repoRoot 'app\client\projects\inference\.cache\compiler.bin'),
        (Join-Path $repoRoot 'app\client\dist\client-angular\index.html'),
        (Join-Path $repoRoot 'app\client\out-tsc\spec\test.js'),
        (Join-Path $repoRoot 'app\client\coverage\client\index.html'),
        (Join-Path $repoRoot 'app\desktop\src-tauri\app\generated.bin'),
        (Join-Path $repoRoot 'app\desktop\build\tauri-cpu-release.json'),
        (Join-Path $repoRoot '.coverage.worker-123'),
        (Join-Path $appDataRoot 'runtimes\cache\python\bytecode.pyc'),
        (Join-Path $appDataRoot 'nltk\corpora\punkt.zip'),
        (Join-Path $appDataRoot 'runtimes\.staging-0123456789abcdef0123456789abcdef\runtime-manifest.json')
    )
    foreach ($path in $expectedCaches) { Write-FixtureFile -Path $path }
    Write-FixtureFile -Path (Join-Path $repoRoot 'runtimes\cache\.gitkeep')

    $persistentFiles = @(
        (Join-Path $resourceRoot 'database.db'),
        (Join-Path $resourceRoot 'checkpoints\saved.xreportckpt'),
        (Join-Path $resourceRoot 'models\huggingface\installed\model.safetensors'),
        (Join-Path $resourceRoot 'logs\backend.log'),
        (Join-Path $resourceRoot 'exports\report.xreportckpt'),
        (Join-Path $appDataRoot '.env'),
        (Join-Path $repoRoot 'settings\.env'),
        (Join-Path $repoRoot 'settings\.env.example'),
        (Join-Path $repoRoot 'settings\.env.production'),
        (Join-Path $repoRoot 'settings\storage-location.json'),
        (Join-Path $repoRoot 'app\server\assets\QA\validation-report.json')
    )
    foreach ($path in $persistentFiles) { Write-FixtureFile -Path $path }
    $pointerPath = Join-Path $appDataRoot 'storage-location.json'
    [IO.File]::WriteAllText($pointerPath, (@{ storage_root = $resourceRoot } | ConvertTo-Json -Compress))
    $resolvedPointerRoot = Resolve-XReportStoragePointer -BootstrapRoot $appDataRoot
    Assert-True ([IO.Path]::GetFullPath($resolvedPointerRoot) -eq [IO.Path]::GetFullPath($resourceRoot)) 'External storage pointer did not resolve to its configured resource root.'
    $persistentFiles += $pointerPath
    $temporaryFiles = @(
        (Join-Path $resourceRoot 'exports\.export-abandoned\partial.bin'),
        (Join-Path $resourceRoot 'checkpoints\.import-upload-abandoned.xreportckpt'),
        (Join-Path $resourceRoot 'models\huggingface\metadata\metadata.0123456789abcdef0123456789abcdef.tmp'),
        (Join-Path $repoRoot 'settings\.env.0123456789abcdef0123456789abcdef.tmp'),
        (Join-Path $repoRoot 'settings\.storage-0123456789abcdef0123456789abcdef.tmp'),
        (Join-Path $appDataRoot 'state\.desktop-ready.json.0123456789abcdef.tmp')
    )
    foreach ($path in $temporaryFiles) { Write-FixtureFile -Path $path }

    $targets = @(Get-XReportCacheTargets -RepoRoot $repoRoot -LocalAppData $localAppData -ResourceRoots @($resourceRoot))
    $targetPaths = @($targets | ForEach-Object { [IO.Path]::GetFullPath($_.Path) })
    foreach ($path in $expectedCaches + $temporaryFiles) {
        $expectedFile = if ((Get-Item -LiteralPath $path -Force).PSIsContainer) { [IO.Path]::GetFullPath($path) } else { [IO.Path]::GetFullPath($path) }
        Assert-True ($targetPaths -contains $expectedFile -or @($targetPaths | Where-Object { $expectedFile.StartsWith($_.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0) "Cache target was not discovered: $expectedFile"
    }
    Assert-True (-not ($targetPaths | Where-Object { $_ -match '\\logs(?:\\|$)' }).Count) 'Cache cleanup must not target logs.'

    $allowedRoots = @($repoRoot, $appDataRoot, $resourceRoot)
    $plan = New-XReportCleanupPlan -Targets $targets -AllowedRoots $allowedRoots
    $dry = Invoke-XReportCleanupPlan -Plan $plan -DryRun
    Assert-True ($dry.Planned.Count -gt 0) 'Dry run did not enumerate deletion candidates.'
    foreach ($path in $expectedCaches + $persistentFiles + $temporaryFiles) {
        Assert-True (Test-Path -LiteralPath $path) "Dry run changed fixture path: $path"
    }

    $result = Invoke-XReportCleanupPlan -Plan $plan
    Assert-True ($result.Failed.Count -eq 0) 'Ordinary fixture cleanup reported failures.'
    foreach ($path in $expectedCaches + $temporaryFiles) {
        Assert-True (-not (Test-Path -LiteralPath $path)) "Expected cache/temp path remains: $path"
    }
    foreach ($path in $persistentFiles) { Assert-True (Test-Path -LiteralPath $path) "Persistent path was removed: $path" }
    Assert-True (Test-Path -LiteralPath (Join-Path $repoRoot 'runtimes\cache\.gitkeep')) 'Cache cleanup removed the sentinel.'
    $pointerAfterCacheCleanup = Resolve-XReportStoragePointer -BootstrapRoot $appDataRoot
    Assert-True ([IO.Path]::GetFullPath($pointerAfterCacheCleanup) -eq [IO.Path]::GetFullPath($resourceRoot)) 'Cache cleanup changed the external storage pointer.'

    $repeatPlan = New-XReportCleanupPlan -Targets $targets -AllowedRoots $allowedRoots
    $repeat = Invoke-XReportCleanupPlan -Plan $repeatPlan
    Assert-True ($repeat.Failed.Count -eq 0) 'Repeated cleanup failed on missing cache paths.'

    $perAppRoot = Join-Path $fixtureRoot 'per-app-external-resources'
    $studioDataRoot = Join-Path $perAppRoot 'studio'
    $dataPointerPayload = @{ storage_root = $perAppRoot } | ConvertTo-Json -Compress
    [IO.File]::WriteAllText($pointerPath, $dataPointerPayload)
    $resolvedDataRoot = Resolve-XReportStoragePointer -BootstrapRoot $appDataRoot
    Assert-True ([IO.Path]::GetFullPath($resolvedDataRoot) -eq [IO.Path]::GetFullPath($perAppRoot)) 'Per-app fixture did not follow the external storage pointer.'
    $inferenceDataFiles = @(
        (Join-Path $perAppRoot 'database.db'),
        (Join-Path $perAppRoot 'database.db-wal'),
        (Join-Path $perAppRoot 'checkpoints\saved.xreportckpt'),
        (Join-Path $perAppRoot 'exports\saved.xreportckpt'),
        (Join-Path $perAppRoot 'logs\inference.log'),
        (Join-Path $perAppRoot 'models\installed\model.safetensors'),
        (Join-Path $perAppRoot 'templates\custom.json'),
        (Join-Path $perAppRoot 'validation_receipts\model.json')
    )
    $studioDataFiles = @(
        (Join-Path $studioDataRoot 'database.db'),
        (Join-Path $studioDataRoot 'checkpoints\trained.xreportckpt'),
        (Join-Path $studioDataRoot 'exports\trained.xreportckpt'),
        (Join-Path $studioDataRoot 'logs\studio.log')
    )
    $perAppSettings = @((Join-Path $perAppRoot '.env'), (Join-Path $perAppRoot 'storage-location.json'))
    foreach ($path in $inferenceDataFiles + $studioDataFiles + $perAppSettings) { Write-FixtureFile -Path $path }
    $inferenceTargets = @(Get-XReportApplicationDataTargets -ResourceRoots @($resolvedDataRoot) -App Xreport)
    $inferencePlan = New-XReportCleanupPlan -Targets $inferenceTargets -AllowedRoots @($perAppRoot)
    $inferenceDryRun = Invoke-XReportCleanupPlan -Plan $inferencePlan -DryRun
    Assert-True ($inferenceDryRun.Planned.Count -gt 0) 'Inference data dry run did not list persistent data targets.'
    foreach ($path in $inferenceDataFiles + $studioDataFiles + $perAppSettings) { Assert-True (Test-Path -LiteralPath $path) "Inference data dry run changed fixture path: $path" }
    $inferenceRemoval = Invoke-XReportCleanupPlan -Plan $inferencePlan
    Assert-True ($inferenceRemoval.Failed.Count -eq 0) 'Inference data removal reported failures.'
    foreach ($path in $inferenceDataFiles) { Assert-True (-not (Test-Path -LiteralPath $path)) "Inference data removal left app data behind: $path" }
    foreach ($path in $studioDataFiles + $perAppSettings) { Assert-True (Test-Path -LiteralPath $path) "Inference data removal crossed an app/settings boundary: $path" }
    $studioTargets = @(Get-XReportApplicationDataTargets -ResourceRoots @($resolvedDataRoot) -App Studio)
    Assert-True (-not @($studioTargets | Where-Object { $_.Path -match '\\models(?:\\|$)' }).Count) 'Studio data targets included the shared models folder.'
    $studioRemoval = Invoke-XReportCleanupPlan -Plan (New-XReportCleanupPlan -Targets $studioTargets -AllowedRoots @($perAppRoot))
    Assert-True ($studioRemoval.Failed.Count -eq 0) 'Studio data removal reported failures.'
    foreach ($path in $studioDataFiles) { Assert-True (-not (Test-Path -LiteralPath $path)) "Studio data removal left app data behind: $path" }
    foreach ($path in $perAppSettings) { Assert-True (Test-Path -LiteralPath $path) "Studio data removal removed shared storage settings: $path" }

    $launcherProbeRoot = Join-Path $fixtureRoot 'launcher-dry-run'
    $launcherLocalAppData = Join-Path $launcherProbeRoot 'profile\AppData\Local'
    $launcherPointerRoot = Join-Path $launcherLocalAppData 'XREPORT\data'
    $launcherSourceRoot = Join-Path $launcherProbeRoot 'source-resources'
    $launcherPackagedRoot = Join-Path $launcherProbeRoot 'packaged-resources'
    foreach ($path in @($launcherPointerRoot, $launcherSourceRoot, $launcherPackagedRoot)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
    $launcherPersistentFiles = @(
        (Join-Path $launcherSourceRoot 'database.db'),
        (Join-Path $launcherSourceRoot 'models\installed\source.bin'),
        (Join-Path $launcherPackagedRoot 'database.db'),
        (Join-Path $launcherPackagedRoot 'models\installed\packaged.bin')
    )
    foreach ($path in $launcherPersistentFiles) { Write-FixtureFile -Path $path }
    [IO.File]::WriteAllText((Join-Path $launcherPointerRoot 'storage-location.json'), (@{ storage_root = $launcherPackagedRoot } | ConvertTo-Json -Compress))
    $launcherScriptPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\start_on_windows.ps1'))
    $oldLauncherResources = $env:XREPORT_RESOURCES_DIR
    $oldLauncherLocalAppData = $env:LOCALAPPDATA
    $origin = Get-Location
    try {
        $env:XREPORT_RESOURCES_DIR = $launcherSourceRoot
        $env:LOCALAPPDATA = $launcherLocalAppData
        Set-Location $env:TEMP
        $launcherOutput = & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $launcherScriptPath -Action RemoveAllData -App Xreport -DryRun 2>&1 | Out-String
        $launcherExitCode = $LASTEXITCODE
        if ($launcherExitCode -ne 0) { throw "RemoveAllData launcher dry run failed: $launcherOutput" }
    } finally {
        Set-Location $origin
        $env:XREPORT_RESOURCES_DIR = $oldLauncherResources
        $env:LOCALAPPDATA = $oldLauncherLocalAppData
    }
    Assert-True ($launcherOutput -match [regex]::Escape($launcherSourceRoot) -and $launcherOutput -match [regex]::Escape($launcherPackagedRoot)) 'Launcher data dry run did not list both configured resource roots.'
    Assert-True ($launcherOutput -match 'Shared XREPORT Inference and Studio models') 'Launcher data dry run did not flag shared model removal.'
    foreach ($path in $launcherPersistentFiles + (Join-Path $launcherPointerRoot 'storage-location.json')) { Assert-True (Test-Path -LiteralPath $path) "Launcher data dry run changed fixture path: $path" }

    $outsideMarker = Join-Path $outsideRoot 'keep.txt'
    Write-FixtureFile -Path $outsideMarker
    $traversalPlan = New-XReportCleanupPlan -Targets @([pscustomobject]@{ Path = (Join-Path $repoRoot '..\outside'); Label = 'Traversal fixture'; KeepRoot = $false; PreserveNames = @() }) -AllowedRoots @($repoRoot)
    Assert-True (@($traversalPlan.InitialResults | Where-Object Status -eq 'Failed').Count -eq 1) 'Traversal target was not rejected.'
    Assert-True (Test-Path -LiteralPath $outsideMarker) 'Traversal validation changed an outside file.'

    $missingPlan = New-XReportCleanupPlan -Targets @([pscustomobject]@{ Path = (Join-Path $repoRoot 'missing\cache'); Label = 'Missing fixture'; KeepRoot = $false; PreserveNames = @() }) -AllowedRoots @($repoRoot)
    $missingResult = Invoke-XReportCleanupPlan -Plan $missingPlan
    Assert-True (@($missingResult.Skipped | Where-Object Message -eq 'Target does not exist.').Count -eq 1) 'Missing target was not reported as skipped.'
    $scanFailurePath = Join-Path $repoRoot 'scan-failure\keep.bin'
    Write-FixtureFile -Path $scanFailurePath
    $scanFailureTarget = [pscustomobject]@{ Path = (Split-Path -Parent $scanFailurePath); Label = 'Unreadable scan fixture'; KeepRoot = $false; PreserveNames = @(); PlanError = 'Simulated enumeration denial.' }
    $scanFailureResult = Invoke-XReportCleanupPlan -Plan (New-XReportCleanupPlan -Targets @($scanFailureTarget) -AllowedRoots @($repoRoot))
    Assert-True ($scanFailureResult.Failed.Count -eq 1 -and (Test-Path -LiteralPath $scanFailurePath)) 'A failed scan was not reported without deleting its directory.'

    $msiEntries = @(
        [pscustomobject]@{ DisplayName = 'XREPORT Inference CPU'; Publisher = 'CTCycle'; UninstallString = 'MsiExec.exe /I{11111111-1111-1111-1111-111111111111}'; InstallLocation = 'C:\Program Files\XREPORT CPU' },
        [pscustomobject]@{ DisplayName = 'XREPORT Inference CUDA'; Publisher = 'CTCycle'; UninstallString = 'MsiExec.exe /X{22222222-2222-2222-2222-222222222222}'; InstallLocation = 'C:\Program Files\XREPORT CUDA' },
        [pscustomobject]@{ DisplayName = 'Other Product'; Publisher = 'CTCycle'; UninstallString = 'MsiExec.exe /X{33333333-3333-3333-3333-333333333333}'; InstallLocation = '' },
        [pscustomobject]@{ DisplayName = 'XREPORT Inference CPU'; Publisher = 'Unrelated'; UninstallString = 'MsiExec.exe /X{44444444-4444-4444-4444-444444444444}'; InstallLocation = '' },
        [pscustomobject]@{ DisplayVersion = '1.0'; PSChildName = '{55555555-5555-5555-5555-555555555555}' }
    )
    $products = @(Select-XReportMsiProducts -RegistryEntries $msiEntries)
    Assert-True ($products.Count -eq 2) 'MSI discovery accepted an unrelated product or missed an XREPORT variant.'
    $startCalls = [Collections.Generic.List[object]]::new()
    $mockMsiStart = { param($filePath, $arguments) [void]$startCalls.Add([pscustomobject]@{ FilePath = $filePath; Arguments = @($arguments) }); [pscustomobject]@{ ExitCode = 0 } }.GetNewClosure()
    $msiOutcome = Invoke-XReportMsiUninstall -Product $products[0] -StartProcess $mockMsiStart
    Assert-True ($msiOutcome.Success -and $startCalls.Count -eq 1) 'Mock MSI uninstall did not complete through the injectable process boundary.'
    Assert-True ($startCalls[0].FilePath -eq 'msiexec.exe' -and $startCalls[0].Arguments -contains '/x' -and $startCalls[0].Arguments -contains $products[0].ProductCode) 'Mock MSI uninstall did not use the selected product code and uninstall verb.'
    $mockMsiFail = { param($filePath, $arguments) [pscustomobject]@{ ExitCode = 1603 } }
    Assert-True (-not (Invoke-XReportMsiUninstall -Product $products[1] -StartProcess $mockMsiFail).Success) 'Mock MSI failure was not reported.'

    $uninstallRepo = Join-Path $fixtureRoot 'uninstall-repository'
    $uninstallLocal = Join-Path $fixtureRoot 'uninstall-profile\AppData\Local'
    $uninstallData = Join-Path $uninstallLocal 'XREPORT\data'
    $uninstallResources = Join-Path $fixtureRoot 'uninstall-resources'
    $portableRoot = Join-Path $fixtureRoot 'portable'
    $portableExe = Join-Path $portableRoot 'XREPORT-v4.0.0-windows-x64-cpu-portable.exe'
    foreach ($path in @($uninstallRepo, $uninstallData, $uninstallResources, $portableRoot)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
    $uninstallPersistent = @(
        (Join-Path $uninstallRepo 'runtimes\.gitkeep'),
        (Join-Path $uninstallRepo 'app\client\projects\inference\src\main.ts'),
        (Join-Path $uninstallResources 'database.db'),
        (Join-Path $uninstallResources 'checkpoints\saved.xreportckpt'),
        (Join-Path $uninstallResources 'models\huggingface\installed\model.safetensors'),
        (Join-Path $uninstallData 'storage-location.json'),
        (Join-Path $portableRoot 'keep.txt')
    )
    foreach ($path in $uninstallPersistent) { Write-FixtureFile -Path $path }
    $uninstallDisposable = @(
        (Join-Path $uninstallRepo 'runtimes\python\python.exe'),
        (Join-Path $uninstallRepo 'app\server\xreport\__pycache__\entry.pyc'),
        (Join-Path $uninstallRepo 'app\client\node_modules\.vite\deps\cache.json'),
        (Join-Path $uninstallRepo 'app\client\.angular\cache\cache.json'),
        (Join-Path $uninstallData 'runtimes\runtime\python.exe'),
        (Join-Path $uninstallData 'runtimes\.staging-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\partial.bin'),
        (Join-Path $uninstallData 'state\desktop-ready.json'),
        (Join-Path $uninstallData 'logs\backend.log'),
        (Join-Path $uninstallResources 'exports\.export-abandoned\partial.bin'),
        (Join-Path $uninstallRepo 'release\XREPORT-v4.0.0-windows-x64-cpu-portable.exe'),
        $portableExe
    )
    foreach ($path in $uninstallDisposable) { Write-FixtureFile -Path $path }
    $uninstallLogTarget = [pscustomobject]@{ Path = (Join-Path $uninstallData 'logs\backend.log'); Label = 'Application log'; KeepRoot = $false; PreserveNames = @() }
    $uninstallTargets = @(Get-XReportUninstallTargets -RepoRoot $uninstallRepo -LocalAppData $uninstallLocal -ResourceRoots @($uninstallResources) -AdditionalTargets @($uninstallLogTarget) -PortableInstallPaths @($portableExe))
    $uninstallPlan = New-XReportCleanupPlan -Targets $uninstallTargets -AllowedRoots @($uninstallRepo, $uninstallData, $uninstallResources, $portableRoot)
    $uninstallDryRun = Invoke-XReportCleanupPlan -Plan $uninstallPlan -DryRun
    Assert-True ($uninstallDryRun.Planned.Count -gt 0) 'Uninstall dry run did not plan disposable fixture paths.'
    foreach ($path in $uninstallPersistent + $uninstallDisposable) { Assert-True (Test-Path -LiteralPath $path) "Uninstall dry run changed fixture path: $path" }
    $uninstallResult = Invoke-XReportCleanupPlan -Plan $uninstallPlan
    Assert-True ($uninstallResult.Failed.Count -eq 0) 'Fixture uninstall reported deletion failures.'
    foreach ($path in $uninstallDisposable) { Assert-True (-not (Test-Path -LiteralPath $path)) "Uninstall left disposable path behind: $path" }
    foreach ($path in $uninstallPersistent) { Assert-True (Test-Path -LiteralPath $path) "Uninstall removed persistent path: $path" }
    try {
        $null = Get-XReportUninstallTargets -RepoRoot $uninstallRepo -PortableInstallPaths @((Join-Path $portableRoot 'unrelated.exe'))
        throw 'Uninstaller accepted an unrelated explicitly named executable.'
    } catch {
        Assert-True ($_.Exception.Message -like 'PortableInstallPath must name a versioned XREPORT portable executable:*') 'Unrelated portable executable failed for an unexpected reason.'
    }

    $lockedPath = Join-Path $repoRoot 'locked\held.bin'
    Write-FixtureFile -Path $lockedPath
    $lockedStream = [IO.File]::Open($lockedPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
    $lockedPlan = New-XReportCleanupPlan -Targets @([pscustomobject]@{ Path = (Split-Path -Parent $lockedPath); Label = 'Locked fixture'; KeepRoot = $false; PreserveNames = @() }) -AllowedRoots @($repoRoot)
    $lockedResult = Invoke-XReportCleanupPlan -Plan $lockedPlan
    Assert-True ($lockedResult.Failed.Count -gt 0) 'Locked file failure was not reported.'
    $lockedStream.Dispose(); $lockedStream = $null
    $retry = Invoke-XReportCleanupPlan -Plan (New-XReportCleanupPlan -Targets @([pscustomobject]@{ Path = (Split-Path -Parent $lockedPath); Label = 'Unlocked fixture'; KeepRoot = $false; PreserveNames = @() }) -AllowedRoots @($repoRoot))
    Assert-True ($retry.Failed.Count -eq 0 -and -not (Test-Path -LiteralPath $lockedPath)) 'Cleanup did not recover after releasing a locked file.'

    New-Item -ItemType Directory -Path $aclTarget -Force | Out-Null
    Write-FixtureFile -Path (Join-Path $aclTarget 'denied.bin')
    try {
        $aclSnapshot = Get-Acl -LiteralPath $aclTarget
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent().User
        $rights = [Security.AccessControl.FileSystemRights]::ReadAndExecute -bor [Security.AccessControl.FileSystemRights]::Delete -bor [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles
        $denyRule = New-Object Security.AccessControl.FileSystemAccessRule($identity, $rights, 'ContainerInherit,ObjectInherit', 'None', 'Deny')
        $deniedAcl = Get-Acl -LiteralPath $aclTarget
        $deniedAcl.AddAccessRule($denyRule)
        Set-Acl -LiteralPath $aclTarget -AclObject $deniedAcl
    } catch {
        if ($null -eq $aclSnapshot) { throw }
        Write-Warning "ACL denial fixture was unavailable in this host context: $($_.Exception.Message)"
        Set-Acl -LiteralPath $aclTarget -AclObject $aclSnapshot -ErrorAction SilentlyContinue
        $aclSnapshot = $null
    }
    if ($null -ne $aclSnapshot) {
        try {
            $aclPlan = New-XReportCleanupPlan -Targets @([pscustomobject]@{ Path = $aclTarget; Label = 'ACL-denied fixture'; KeepRoot = $false; PreserveNames = @() }) -AllowedRoots @($fixtureRoot)
            $aclResult = Invoke-XReportCleanupPlan -Plan $aclPlan
            Assert-True ($aclResult.Failed.Count -gt 0) 'ACL-denied enumeration did not report a failure.'
        } finally {
            Set-Acl -LiteralPath $aclTarget -AclObject $aclSnapshot -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path -LiteralPath $aclTarget) {
        $aclRetry = Invoke-XReportCleanupPlan -Plan (New-XReportCleanupPlan -Targets @([pscustomobject]@{ Path = $aclTarget; Label = 'ACL-restored fixture'; KeepRoot = $false; PreserveNames = @() }) -AllowedRoots @($fixtureRoot))
        Assert-True ($aclRetry.Failed.Count -eq 0 -and -not (Test-Path -LiteralPath $aclTarget)) 'Cleanup did not recover after restoring the ACL.'
    }

    $junctionTarget = Join-Path $outsideRoot 'junction-target'
    New-Item -ItemType Directory -Path $junctionTarget -Force | Out-Null
    Write-FixtureFile -Path (Join-Path $junctionTarget 'protected.txt')
    $junctionCreated = $false
    try {
        New-Item -ItemType Junction -Path $junctionPath -Target $junctionTarget -ErrorAction Stop | Out-Null
        $junctionCreated = $true
    } catch {
        Write-Warning "Junction fixture could not be created by this PowerShell host: $($_.Exception.Message)"
    }
    if ($junctionCreated) {
        $junctionResult = Invoke-XReportCleanupPlan -Plan (New-XReportCleanupPlan -Targets @([pscustomobject]@{ Path = $junctionPath; Label = 'Junction fixture'; KeepRoot = $false; PreserveNames = @() }) -AllowedRoots @($repoRoot))
        Assert-True ($junctionResult.Skipped.Count -gt 0) 'Junction was not reported as skipped.'
        Assert-True (Test-Path -LiteralPath (Join-Path $junctionTarget 'protected.txt')) 'Junction cleanup followed the link outside the allowed root.'
        [IO.Directory]::Delete($junctionPath, $false)
    }

    Write-Output 'PASS: launcher cleanup module fixture checks.'
} finally {
    if ($null -ne $lockedStream) { $lockedStream.Dispose() }
    if (Test-Path -LiteralPath $junctionPath) { try { [IO.Directory]::Delete($junctionPath, $false) } catch {} }
    if (Test-Path -LiteralPath $fixtureRoot) { Remove-Item -LiteralPath $fixtureRoot -Force -Recurse -ErrorAction SilentlyContinue }
}
