[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda')]
    [string]$Variant = 'cpu',
    [string]$Version = '3.1.0',
    [string]$ReleaseRoot,
    [string]$ValidationRoot,
    [string]$SeedResourceRoot,
    [string]$FixtureRoot,
    [string]$Output,
    [ValidateSet('portable', 'msi')][string]$PackageKind = 'portable',
    [string]$ExecutablePath,
    [string]$NativeOutput,
    [switch]$ExerciseTrainingUiSubmitCancel
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
if ([string]::IsNullOrWhiteSpace($ReleaseRoot)) {
    $ReleaseRoot = Join-Path $repoRoot 'release'
}
if ([string]::IsNullOrWhiteSpace($SeedResourceRoot)) {
    $SeedResourceRoot = Join-Path $repoRoot 'runtimes\cache\release-validation-20260925\resources'
}
if ([string]::IsNullOrWhiteSpace($FixtureRoot)) {
    $FixtureRoot = Join-Path $repoRoot 'assets\QA\validation_campaign\release-3.2.0-20261017\fixtures'
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $repoRoot "assets\QA\desktop\packaged-active-training-$Variant-$Version.json"
}
$ReleaseRoot = [IO.Path]::GetFullPath($ReleaseRoot)
$SeedResourceRoot = [IO.Path]::GetFullPath($SeedResourceRoot)
$FixtureRoot = [IO.Path]::GetFullPath($FixtureRoot)
$Output = [IO.Path]::GetFullPath($Output)
$ownsValidationRoot = [string]::IsNullOrWhiteSpace($ValidationRoot)
if ($ownsValidationRoot) {
    $ValidationRoot = Join-Path $repoRoot "runtimes\cache\packaged-active-training-$PID-$Variant-$Version"
}
$ValidationRoot = [IO.Path]::GetFullPath($ValidationRoot)
$portable = Join-Path $ReleaseRoot "XREPORT-v$Version-windows-x64-$Variant-portable.exe"
$packagedExecutable = if ([string]::IsNullOrWhiteSpace($ExecutablePath)) { $portable } else { [IO.Path]::GetFullPath($ExecutablePath) }
if ($PackageKind -eq 'msi' -and [string]::IsNullOrWhiteSpace($ExecutablePath)) { throw 'ExecutablePath is required for an installed MSI package.' }
$csvPath = Get-ChildItem -LiteralPath $FixtureRoot -File -Filter '*.csv' | Select-Object -First 1
$imageFolder = Join-Path $FixtureRoot 'images'
if (-not (Test-Path -LiteralPath $packagedExecutable -PathType Leaf)) { throw "Packaged executable is missing: $packagedExecutable" }
if ($null -eq $csvPath) { throw "Fixture CSV is missing under $FixtureRoot" }
if (-not (Test-Path -LiteralPath $imageFolder -PathType Container)) { throw "Fixture image folder is missing: $imageFolder" }
foreach ($relative in @('models\XRAYEncoder', 'models\tokenizers')) {
    if (-not (Test-Path -LiteralPath (Join-Path $SeedResourceRoot $relative) -PathType Container)) {
        throw "Seed resource directory is missing: $(Join-Path $SeedResourceRoot $relative)"
    }
}

$validationRootPrefixes = @()
$validationRootPrefixes += ([IO.Path]::GetFullPath((Join-Path $repoRoot 'runtimes\cache'))).TrimEnd('\') + '\'
try {
    $currentUserSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $profileKey = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList\$currentUserSid"
    $currentUserProfile = [Environment]::ExpandEnvironmentVariables([string](Get-ItemProperty -LiteralPath $profileKey -ErrorAction Stop).ProfileImagePath)
    $currentUserLocalAppData = Join-Path $currentUserProfile 'AppData\Local'
    $validationRootPrefixes += ([IO.Path]::GetFullPath((Join-Path $currentUserLocalAppData 'xreport-release-validation'))).TrimEnd('\') + '\'
} catch {
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $validationRootPrefixes += ([IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'xreport-release-validation'))).TrimEnd('\') + '\'
    }
}
$validationRootAllowed = $false
foreach ($validationRootPrefix in $validationRootPrefixes) {
    $rootPrefixMatches = $ValidationRoot.StartsWith($validationRootPrefix, [StringComparison]::OrdinalIgnoreCase)
    if ($rootPrefixMatches) {
        $validationRootAllowed = $true
        break
    }
}
if (-not $validationRootAllowed) {
    throw "ValidationRoot must remain under the repository runtimes/cache directory or the current user's task-specific local-app-data validation directory: $ValidationRoot"
}
if (Test-Path -LiteralPath $ValidationRoot) {
    throw "ValidationRoot already exists; use a new task-owned path: $ValidationRoot"
}

$dataRoot = Join-Path $ValidationRoot 'localappdata\XREPORT\data'
$stateRoot = Join-Path $dataRoot 'state'
$sessionPath = Join-Path $stateRoot 'desktop-session.json'
$readyPath = Join-Path $stateRoot 'desktop-ready.json'
$datasetName = "s51-packaged-$PID"
$processedName = "$datasetName-processed"
$checkpointName = "$datasetName-checkpoint"
$desktopToken = $null
$process = $null
$launcherProcess = $null
$backendPid = $null
$port = $null
$webSession = $null
$apiHeaders = @{}
$previousLocalAppData = $env:LOCALAPPDATA
$receipt = [ordered]@{
    schema_version = 'xreport-packaged-active-training-close-reopen-v1'
    observed_utc = [DateTime]::UtcNow.ToString('o')
    repository_head = (git -C $repoRoot rev-parse HEAD).Trim()
    dirty_tree = [bool](@(git -C $repoRoot status --porcelain).Count)
    variant = $Variant
    version = $Version
    package_kind = $PackageKind
    executable = $packagedExecutable
    fixture = [ordered]@{ root = $FixtureRoot; csv = $csvPath.Name; dataset = $datasetName; processed_dataset = $processedName; checkpoint = $checkpointName; synthetic = $true }
    seed = [ordered]@{ resource_root = $SeedResourceRoot; encoder = 'models/XRAYEncoder'; tokenizer = 'models/tokenizers' }
    start = $null
    native_acceptance = $null
    training_ui_submit_cancel = $null
    training = $null
    close_during_training = $null
    reopen = $null
    cleanup = $null
    stage = 'initializing'
    passed = $false
}

function Wait-Path {
    param([Parameter(Mandatory = $true)][string]$Path, [int]$TimeoutSeconds = 120)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        if (Test-Path -LiteralPath $Path -PathType Leaf) { return $true }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Wait-JobTerminal {
    param([Parameter(Mandatory = $true)][string]$BaseUrl, [Parameter(Mandatory = $true)][string]$JobId, [int]$TimeoutSeconds = 240)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $last = $null
    do {
        try {
            $last = Invoke-RestMethod -Uri "$BaseUrl/api/jobs/$JobId" -WebSession $script:webSession -TimeoutSec 10
        } catch {
            throw "Job polling failed for ${BaseUrl}/api/jobs/${JobId}: $($_.Exception.Message)"
        }
        if ($last.status -in @('completed', 'failed', 'cancelled')) { return $last }
        Start-Sleep -Seconds 1
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Job $JobId did not reach a terminal state within $TimeoutSeconds seconds."
}

function Wait-BackendGone {
    param([int]$BackendProcessId, [int]$Port, [int]$TimeoutSeconds = 30)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $processGone = $null -eq (Get-Process -Id $BackendProcessId -ErrorAction SilentlyContinue)
        $listenerGone = $null -eq (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
        if ($processGone -and $listenerGone) { return $true }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Close-PackagedProcess {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Target,
        [switch]$AllowForcedKill
    )
    $forcedKill = $false
    if (-not $Target.HasExited) {
        try { $Target.CloseMainWindow() | Out-Null } catch { }
        try { [void]$Target.WaitForExit(30000) } catch { }
    }
    $Target.Refresh()
    if (-not $Target.HasExited -and $AllowForcedKill) {
        Stop-Process -Id $Target.Id -Force -ErrorAction SilentlyContinue
        try { [void]$Target.WaitForExit(10000) } catch { }
        $forcedKill = $true
    }
    $Target.Refresh()
    return [ordered]@{
        graceful_exit = [bool]($Target.HasExited -and -not $forcedKill)
        app_exited = [bool]$Target.HasExited
        forced_kill = [bool]$forcedKill
    }
}

function Invoke-JsonApi {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('Get', 'Post', 'Patch')][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [object]$Body
    )
    try {
        if ($null -eq $Body) { return Invoke-RestMethod -Method $Method -Uri $Uri -WebSession $script:webSession -TimeoutSec 30 }
        $json = $Body | ConvertTo-Json -Depth 10
        return Invoke-RestMethod -Method $Method -Uri $Uri -WebSession $script:webSession -ContentType 'application/json' -Body $json -TimeoutSec 30
    } catch {
        throw "JSON API $Method $Uri failed: $($_.Exception.Message)"
    }
}

function Start-Packaged {
    $script:launcherProcess = Start-Process -FilePath $packagedExecutable -WorkingDirectory $ValidationRoot -PassThru
    $script:process = $script:launcherProcess
    if (-not (Wait-Path -Path $sessionPath -TimeoutSeconds 180)) { throw 'Packaged session contract was not written.' }
    $session = Get-Content -LiteralPath $sessionPath -Raw | ConvertFrom-Json
    $script:backendPid = [int]$session.pid
    $script:port = ([Uri]$session.bootstrap_url).Port
    if (-not (Wait-Path -Path $readyPath -TimeoutSeconds 30)) { throw 'Packaged readiness contract was not written.' }
    $bootstrap = [Uri]$session.bootstrap_url
    $script:webSession = New-Object Microsoft.PowerShell.Commands.WebRequestSession
    $bootstrapResponse = Invoke-WebRequest -UseBasicParsing -Uri $bootstrap -WebSession $script:webSession -TimeoutSec 20
    if ($bootstrapResponse.StatusCode -lt 200 -or $bootstrapResponse.StatusCode -ge 300) {
        throw "Packaged bootstrap did not establish a session (status $($bootstrapResponse.StatusCode))."
    }
    $token = [Uri]::UnescapeDataString(($bootstrap.Query -replace '^\?token=', ''))
    $script:desktopToken = $token
    $script:apiHeaders = @{ 'X-XREPORT-Desktop-Token' = $token }
    $healthResponse = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/api/health" -Headers $script:apiHeaders -WebSession $script:webSession -TimeoutSec 20
    $health = $healthResponse.Content | ConvertFrom-Json
    if ($health.status -ne 'ok' -or $health.runtime_variant -ne $Variant -or $health.version -ne $Version) {
        throw 'Packaged health contract did not match the requested variant/version.'
    }
    $windowProcess = Find-PackagedWindowProcess -LauncherProcessId $script:launcherProcess.Id -TimeoutSeconds 30
    $script:process = $windowProcess
    $safeSession = [ordered]@{
        pid = [int]$session.pid
        variant = [string]$session.variant
        version = [string]$session.version
        bootstrap_path = $bootstrap.AbsolutePath
    }
    return [ordered]@{ pid = [int]$script:launcherProcess.Id; window_pid = [int]$windowProcess.Id; backend_pid = $backendPid; port = $port; health = $health; session = $safeSession }
}

function Find-PackagedWindowProcess {
    param([Parameter(Mandatory = $true)][int]$LauncherProcessId, [int]$TimeoutSeconds = 30)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $records = @(Get-CimInstance -ClassName Win32_Process -ErrorAction SilentlyContinue)
        $processIds = [System.Collections.Generic.HashSet[int]]::new()
        [void]$processIds.Add($LauncherProcessId)
        $expanded = $true
        while ($expanded) {
            $expanded = $false
            foreach ($record in $records) {
                if ($processIds.Contains([int]$record.ParentProcessId) -and $processIds.Add([int]$record.ProcessId)) {
                    $expanded = $true
                }
            }
        }
        foreach ($id in $processIds) {
            $candidate = Get-Process -Id $id -ErrorAction SilentlyContinue
            if ($null -eq $candidate) { continue }
            $candidate.Refresh()
            $candidatePath = [string]$candidate.Path
            if ($candidate.MainWindowHandle -ne 0 -and (
                $candidate.MainWindowTitle -match '(?i)XREPORT.*Radiological Reports' -or
                [string]::Equals($candidatePath, $packagedExecutable, [StringComparison]::OrdinalIgnoreCase)
            )) {
                return $candidate
            }
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "The packaged XREPORT main window was not found among launcher process $LauncherProcessId and its descendants."
}

function Wait-PackagedProcessExit {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$Target, [int]$TimeoutSeconds = 30)
    $Target.Refresh()
    if (-not $Target.HasExited) {
        try { [void]$Target.WaitForExit($TimeoutSeconds * 1000) } catch { }
    }
    $Target.Refresh()
    return [bool]$Target.HasExited
}

try {
    New-Item -ItemType Directory -Path (Join-Path $dataRoot 'models') -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $SeedResourceRoot 'models\XRAYEncoder') -Destination (Join-Path $dataRoot 'models') -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $SeedResourceRoot 'models\tokenizers') -Destination (Join-Path $dataRoot 'models') -Recurse -Force
    New-Item -ItemType Directory -Path (Split-Path -Parent $Output) -Force | Out-Null
    $env:LOCALAPPDATA = Join-Path $ValidationRoot 'localappdata'

    $receipt.start = Start-Packaged
    $receipt.stage = 'session_started'
    $baseUrl = "http://127.0.0.1:$($receipt.start.port)"
    $receipt.stage = 'settings'
    $settings = Invoke-JsonApi -Method Patch -Uri "$baseUrl/api/settings" -Body @{ features = @{ allow_local_filesystem_access = $true } }
    $receipt.stage = 'upload'
    try {
        $uploadResponse = Invoke-WebRequest -Method Post -Uri "$baseUrl/api/upload/dataset" -WebSession $webSession -Form @{ file = Get-Item -LiteralPath $csvPath.FullName } -TimeoutSec 30
    } catch {
        throw "Dataset upload failed for $baseUrl/api/upload/dataset: $($_.Exception.Message)"
    }
    $upload = $uploadResponse.Content | ConvertFrom-Json
    $sourceDatasetName = [string]$upload.dataset_name
    if ([string]::IsNullOrWhiteSpace($sourceDatasetName)) { throw 'Dataset upload did not return a dataset name.' }
    $receipt.fixture.dataset = $sourceDatasetName
    $receipt.stage = 'load'
    $loaded = Invoke-JsonApi -Method Post -Uri "$baseUrl/api/preparation/dataset/load" -Body @{
        upload_id = $upload.upload_id
        image_folder_path = $imageFolder
        sample_size = 1.0
        confirm_unmatched = $false
    }
    if (-not $loaded.success -or $loaded.matched_records -ne 8) { throw "Packaged fixture load did not match all 8 rows: $($loaded | ConvertTo-Json -Compress)" }
    $receipt.stage = 'process'
    $processingStart = Invoke-JsonApi -Method Post -Uri "$baseUrl/api/preparation/dataset/process" -Body @{
        dataset_name = $sourceDatasetName
        custom_name = $processedName
        sample_size = 1.0
        validation_size = 0.25
        tokenizer = 'distilbert-base-uncased'
        max_report_size = 200
    }
    $processing = Wait-JobTerminal -BaseUrl $baseUrl -JobId $processingStart.job_id -TimeoutSeconds 240
    if ($processing.status -ne 'completed') { throw "Packaged dataset processing failed: $($processing | ConvertTo-Json -Compress)" }

    if ($ExerciseTrainingUiSubmitCancel) {
        $receipt.stage = 'native_training_submit_cancel'
        if ([string]::IsNullOrWhiteSpace($NativeOutput)) {
            $nativeStem = [IO.Path]::GetFileNameWithoutExtension($Output)
            $NativeOutput = Join-Path (Split-Path -Parent $Output) "$nativeStem-native.json"
        }
        $nativeOutputPath = [IO.Path]::GetFullPath($NativeOutput)
        $pwsh = (Get-Command pwsh -ErrorAction Stop).Source
        $nativeScript = Join-Path $PSScriptRoot 'validate_native_webview.ps1'
        $nativeArguments = @(
            '-NoProfile', '-File', $nativeScript,
            '-Output', $nativeOutputPath,
            '-Scenario', 'Routes,TrainingSubmitCancel',
            '-StartupUiReadyTimeoutSeconds', '90',
            '-WindowProcessId', [string]$script:process.Id,
            '-BackendPort', [string]$port,
            '-DesktopToken', $desktopToken,
            '-TrainingDatasetName', $processedName,
            '-TrainingDevice', $(if ($Variant -eq 'cuda') { 'cuda' } else { 'cpu' })
        )
        & $pwsh @nativeArguments
        if ($LASTEXITCODE -ne 0) { throw "Native packaged UI validation failed with exit code $LASTEXITCODE." }
        $nativeReceipt = Get-Content -LiteralPath $nativeOutputPath -Raw | ConvertFrom-Json
        $receipt.native_acceptance = [ordered]@{
            output = $nativeOutputPath
            passed = [bool]$nativeReceipt.passed
            requested_scenarios = @($nativeReceipt.requested_scenarios)
            scenarios = $nativeReceipt.scenarios
            screenshots = @($nativeReceipt.screenshots)
        }
        $receipt.training_ui_submit_cancel = $nativeReceipt.scenarios.training_submit_cancel
        if (-not $nativeReceipt.passed -or $nativeReceipt.scenarios.training_submit_cancel.status -ne 'PASS') {
            throw 'Packaged native UI routes or Training submission/cancellation did not pass.'
        }
    }

    $receipt.stage = 'training_start'
    $trainingBody = @{
        dataset_name = $processedName; epochs = 1; batch_size = 1; num_encoders = 1; num_decoders = 1
        embedding_dims = 64; attention_heads = 1; train_temp = 1.0; freeze_img_encoder = $true
        use_img_augmentation = $false; shuffle_with_buffer = $false; shuffle_size = 256; save_checkpoints = $true
        checkpoint_id = $checkpointName; use_device_GPU = ($Variant -eq 'cuda'); device_ID = 0; jit_compile = $false
        jit_backend = 'inductor'; use_mixed_precision = $false; dataloader_workers = 0; prefetch_factor = 1
        pin_memory = $false; persistent_workers = $false; plot_training_metrics = $false; use_scheduler = $false
        target_LR = 0.001; warmup_steps = 1000
    }
    $trainingStart = Invoke-JsonApi -Method Post -Uri "$baseUrl/api/training/start" -Body $trainingBody
    $receipt.stage = 'training_active_probe'
    Start-Sleep -Seconds 3
    $trainingStatus = Invoke-RestMethod -Uri "$baseUrl/api/jobs/$($trainingStart.job_id)" -WebSession $webSession -TimeoutSec 20
    if ($trainingStatus.status -notin @('queued', 'running')) { throw "Training was not active before close: $($trainingStatus | ConvertTo-Json -Compress)" }
    $receipt.training = [ordered]@{ job_id = $trainingStart.job_id; start = $trainingStart; before_close = $trainingStatus; processing_job_id = $processingStart.job_id }

    $receipt.stage = 'close_during_training'
    $firstClose = Close-PackagedProcess -Target $process
    $launcherExited = if ($launcherProcess.Id -eq $process.Id) { [bool]$firstClose.app_exited } else { Wait-PackagedProcessExit -Target $launcherProcess -TimeoutSeconds 30 }
    $firstCleanup = Wait-BackendGone -BackendProcessId $backendPid -Port $port -TimeoutSeconds 45
    $receipt.close_during_training = [ordered]@{ shell_exited = $firstClose.app_exited; launcher_exited = $launcherExited; window_pid = [int]$process.Id; launcher_pid = [int]$launcherProcess.Id; graceful_exit = [bool]($firstClose.graceful_exit -and $launcherExited); forced_kill = $firstClose.forced_kill; backend_and_listener_gone = $firstCleanup; backend_pid = $backendPid; port = $port }
    if (-not $firstClose.graceful_exit -or -not $launcherExited -or $firstClose.forced_kill -or -not $firstCleanup) { throw 'Packaged close did not gracefully remove the shell, launcher, backend, and listener after active training.' }
    $process = $null
    $launcherProcess = $null

    $receipt.stage = 'reopen'
    $receipt.reopen = Start-Packaged
    $reopenBaseUrl = "http://127.0.0.1:$($receipt.reopen.port)"
    $runningAfterReopen = Invoke-JsonApi -Method Get -Uri "$reopenBaseUrl/api/jobs?status=running"
    $reopenedTrainingResponse = Invoke-WebRequest -Uri "$reopenBaseUrl/api/jobs/$($trainingStart.job_id)" -WebSession $webSession -SkipHttpErrorCheck -TimeoutSec 20
    if ($reopenedTrainingResponse.StatusCode -eq 200) {
        $reopenedTraining = $reopenedTrainingResponse.Content | ConvertFrom-Json
    } else {
        $reopenedTraining = [ordered]@{ status_code = [int]$reopenedTrainingResponse.StatusCode; status = 'not_found_after_reopen' }
    }
    $receipt.reopen.running_jobs = $runningAfterReopen
    $receipt.reopen.reopened_training_job_status = $reopenedTraining
    $receipt.reopen.no_running_jobs = @($runningAfterReopen.jobs).Count -eq 0
    $receipt.stage = 'final_cleanup'
    $secondClose = Close-PackagedProcess -Target $process
    $secondLauncherExited = if ($launcherProcess.Id -eq $process.Id) { [bool]$secondClose.app_exited } else { Wait-PackagedProcessExit -Target $launcherProcess -TimeoutSeconds 30 }
    $secondCleanup = Wait-BackendGone -BackendProcessId $backendPid -Port $port -TimeoutSeconds 45
    $receipt.cleanup = [ordered]@{ shell_exited = $secondClose.app_exited; launcher_exited = $secondLauncherExited; window_pid = [int]$process.Id; launcher_pid = [int]$launcherProcess.Id; graceful_exit = [bool]($secondClose.graceful_exit -and $secondLauncherExited); forced_kill = $secondClose.forced_kill; backend_and_listener_gone = $secondCleanup; backend_pid = $backendPid; port = $port }
    $receipt.passed = [bool]($receipt.close_during_training.graceful_exit -and -not $receipt.close_during_training.forced_kill -and $receipt.close_during_training.backend_and_listener_gone -and $receipt.reopen.no_running_jobs -and $secondClose.graceful_exit -and $secondLauncherExited -and -not $secondClose.forced_kill -and $secondCleanup -and (-not $ExerciseTrainingUiSubmitCancel -or $receipt.training_ui_submit_cancel.status -eq 'PASS'))
}
catch {
    $receipt.error = $_.Exception.Message
    if ($process) {
        $failedClose = Close-PackagedProcess -Target $process -AllowForcedKill
        $launcherExited = if ($null -eq $launcherProcess -or $launcherProcess.Id -eq $process.Id) { [bool]$failedClose.app_exited } else { Wait-PackagedProcessExit -Target $launcherProcess -TimeoutSeconds 10 }
        if (-not $launcherExited -and $launcherProcess -and $launcherProcess.Id -ne $process.Id) {
            Stop-Process -Id $launcherProcess.Id -Force -ErrorAction SilentlyContinue
            $launcherExited = Wait-PackagedProcessExit -Target $launcherProcess -TimeoutSeconds 10
        }
        $receipt.cleanup = [ordered]@{ shell_exited = $failedClose.app_exited; launcher_exited = $launcherExited; window_pid = [int]$process.Id; launcher_pid = if ($launcherProcess) { [int]$launcherProcess.Id } else { $null }; graceful_exit = [bool]($failedClose.graceful_exit -and $launcherExited); forced_kill = [bool]($failedClose.forced_kill -or (-not $launcherExited)); backend_and_listener_gone = $false; backend_pid = $backendPid; port = $port }
    }
    if ($backendPid -and $port) {
        $cleanupPass = Wait-BackendGone -BackendProcessId $backendPid -Port $port -TimeoutSeconds 30
        if ($null -eq $receipt.cleanup) { $receipt.cleanup = [ordered]@{ shell_exited = $true; graceful_exit = $null; forced_kill = $false; backend_and_listener_gone = $cleanupPass; backend_pid = $backendPid; port = $port } }
        else { $receipt.cleanup.backend_and_listener_gone = $cleanupPass }
    }
}
finally {
    if ($null -eq $previousLocalAppData) { Remove-Item Env:LOCALAPPDATA -ErrorAction SilentlyContinue } else { $env:LOCALAPPDATA = $previousLocalAppData }
    $receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Output -Encoding utf8
    if ($ownsValidationRoot -and (Test-Path -LiteralPath $ValidationRoot)) {
        Remove-Item -LiteralPath $ValidationRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not $receipt.passed) { throw "Packaged active-training close/reopen validation failed: $($receipt | ConvertTo-Json -Compress)" }
Write-Host "Packaged active-training close/reopen validation passed: $Output"
