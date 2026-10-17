[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('cpu', 'cuda')][string]$Variant,
    [Parameter(Mandatory = $true)][ValidateSet('portable', 'msi')][string]$PackageKind,
    [Parameter(Mandatory = $true)][string]$ExecutablePath,
    [Parameter(Mandatory = $true)][string]$ValidationRoot,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$FixtureImage = (Join-Path (Split-Path -Parent $PSScriptRoot) 'fixtures\images\synthetic_01.png'),
    [string]$ModelRef = 'huggingface:aehrc/cxrmate-multi-tf',
    [string]$ModelRevision = '330721b9aa5bba201a3eb88eba4dd9a6607f3e7a',
    [int]$DownloadTimeoutSeconds = 1800,
    [int]$InferenceTimeoutSeconds = 1800
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..\..')).Path
$ExecutablePath = [IO.Path]::GetFullPath($ExecutablePath)
$ValidationRoot = [IO.Path]::GetFullPath($ValidationRoot)
$Output = [IO.Path]::GetFullPath($Output)
$FixtureImage = [IO.Path]::GetFullPath($FixtureImage)
$validationPrefix = [IO.Path]::GetFullPath((Join-Path $repoRoot 'runtimes\cache')).TrimEnd('\') + '\'
if (-not $ValidationRoot.StartsWith($validationPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "ValidationRoot must stay inside the isolated runtimes/cache directory: $ValidationRoot"
}
if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) { throw "Packaged executable is missing: $ExecutablePath" }
if (-not (Test-Path -LiteralPath $FixtureImage -PathType Leaf)) { throw "Synthetic inference image is missing: $FixtureImage" }
if ($DownloadTimeoutSeconds -lt 60 -or $InferenceTimeoutSeconds -lt 60) { throw 'Job timeouts must be at least 60 seconds.' }

$localAppData = Join-Path $ValidationRoot 'localappdata'
$sessionPath = Join-Path $localAppData 'XREPORT\data\state\desktop-session.json'
$readyPath = Join-Path $localAppData 'XREPORT\data\state\desktop-ready.json'
$previousLocalAppData = $env:LOCALAPPDATA
$launcher = $null
$windowProcess = $null
$backendPid = 0
$port = 0
$webSession = $null
$desktopToken = $null
$baseUrl = $null
$receipt = [ordered]@{
    schema_version = 'xreport-packaged-inference-v1'
    observed_utc = [DateTime]::UtcNow.ToString('o')
    variant = $Variant
    package_kind = $PackageKind
    executable = $ExecutablePath
    source_commit = 'a8d75a0c0d5f02a29c85e9ac58f4837fc3081449'
    fixture_image = [IO.Path]::GetFileName($FixtureImage)
    model_ref = $ModelRef
    model_revision = $ModelRevision
    start = $null
    model_download = $null
    inference = $null
    cleanup = $null
    passed = $false
    error = $null
}

function Invoke-JsonApi {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('Get', 'Post')][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [object]$Body
    )
    if ($null -eq $Body) {
        return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $script:headers -WebSession $script:webSession -TimeoutSec 60
    }
    $json = $Body | ConvertTo-Json -Depth 12
    return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $script:headers -WebSession $script:webSession -ContentType 'application/json' -Body $json -TimeoutSec 60
}

function Wait-JobTerminal {
    param(
        [Parameter(Mandatory = $true)][string]$JobId,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)][string]$JobLabel
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $last = $null
    do {
        $last = Invoke-JsonApi -Method Get -Uri "$script:baseUrl/api/jobs/$JobId"
        if ($last.status -in @('completed', 'failed', 'cancelled')) { return $last }
        Start-Sleep -Seconds 1
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "$JobLabel job $JobId timed out: $($last | ConvertTo-Json -Compress -Depth 8)"
}

function Find-PackagedWindowProcess {
    param([Parameter(Mandatory = $true)][int]$LauncherProcessId, [int]$TimeoutSeconds = 45)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $records = @(Get-CimInstance -ClassName Win32_Process -ErrorAction SilentlyContinue)
        $processIds = [System.Collections.Generic.HashSet[int]]::new()
        [void]$processIds.Add($LauncherProcessId)
        $changed = $true
        while ($changed) {
            $changed = $false
            foreach ($record in $records) {
                if ($processIds.Contains([int]$record.ParentProcessId) -and $processIds.Add([int]$record.ProcessId)) {
                    $changed = $true
                }
            }
        }
        foreach ($id in $processIds) {
            $candidate = Get-Process -Id $id -ErrorAction SilentlyContinue
            if ($candidate -and $candidate.MainWindowHandle -ne 0 -and $candidate.MainWindowTitle -match '(?i)XREPORT.*Radiological Reports') {
                return $candidate
            }
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "The packaged XREPORT window was not found beneath launcher process $LauncherProcessId."
}

function Wait-BackendGone {
    param([int]$ProcessId, [int]$Port, [int]$TimeoutSeconds = 45)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $processGone = $null -eq (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
        $listenerGone = $null -eq (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($processGone -and $listenerGone) { return $true }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

try {
    New-Item -ItemType Directory -Path $localAppData -Force | Out-Null
    foreach ($contractPath in @($sessionPath, $readyPath)) {
        if (Test-Path -LiteralPath $contractPath -PathType Leaf) { Remove-Item -LiteralPath $contractPath -Force }
    }
    $env:LOCALAPPDATA = $localAppData
    $launcher = Start-Process -FilePath $ExecutablePath -WorkingDirectory $ValidationRoot -PassThru
    $sessionDeadline = [DateTime]::UtcNow.AddSeconds(180)
    do {
        if ($launcher.HasExited) { throw "Packaged $Variant $PackageKind application exited during startup with code $($launcher.ExitCode)." }
        if (Test-Path -LiteralPath $sessionPath -PathType Leaf) { break }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $sessionDeadline)
    if (-not (Test-Path -LiteralPath $sessionPath -PathType Leaf)) { throw 'Packaged application did not create desktop-session.json.' }

    $session = Get-Content -LiteralPath $sessionPath -Raw | ConvertFrom-Json
    if ($session.version -ne '3.2.0' -or $session.variant -ne $Variant) { throw 'Desktop session version or variant does not match the package.' }
    $backendPid = [int]$session.pid
    $bootstrap = [Uri]$session.bootstrap_url
    $port = $bootstrap.Port
    $baseUrl = "http://127.0.0.1:$port"
    $webSession = New-Object Microsoft.PowerShell.Commands.WebRequestSession
    $bootstrapResponse = Invoke-WebRequest -UseBasicParsing -Uri $bootstrap -WebSession $webSession -TimeoutSec 30
    if ($bootstrapResponse.StatusCode -lt 200 -or $bootstrapResponse.StatusCode -ge 300) { throw 'Packaged desktop bootstrap did not establish a web session.' }
    $desktopToken = [Uri]::UnescapeDataString(($bootstrap.Query -replace '^\?token=', ''))
    $headers = @{ 'X-XREPORT-Desktop-Token' = $desktopToken }
    $script:headers = $headers
    $script:webSession = $webSession
    $script:baseUrl = $baseUrl
    $health = Invoke-JsonApi -Method Get -Uri "$baseUrl/api/health"
    if ($health.status -ne 'ok' -or $health.runtime_variant -ne $Variant -or $health.version -ne '3.2.0') { throw 'Packaged health contract does not match the expected release variant.' }
    $windowProcess = Find-PackagedWindowProcess -LauncherProcessId $launcher.Id
    $receipt.start = [ordered]@{
        launcher_pid = [int]$launcher.Id
        window_pid = [int]$windowProcess.Id
        backend_pid = $backendPid
        port = $port
        health = $health
        session = [ordered]@{ version = [string]$session.version; variant = [string]$session.variant; pid = $backendPid; bootstrap_path = $bootstrap.AbsolutePath }
    }

    $maintenance = Invoke-JsonApi -Method Post -Uri "$baseUrl/api/inference/models/maintenance" -Body @{
        model_ref = $ModelRef
        action = 'download'
        revision = $ModelRevision
    }
    $maintenanceJob = Wait-JobTerminal -JobId ([string]$maintenance.job_id) -TimeoutSeconds $DownloadTimeoutSeconds -JobLabel 'Model download'
    $receipt.model_download = [ordered]@{
        job_id = [string]$maintenance.job_id
        status = [string]$maintenanceJob.status
        error = $maintenanceJob.error
        result = $maintenanceJob.result
    }
    if ($maintenanceJob.status -ne 'completed') { throw "Pinned model download did not complete: $($maintenanceJob | ConvertTo-Json -Compress -Depth 8)" }

    $form = @{
        model_ref = $ModelRef
        generation_profile = 'deterministic'
        clinical_context = ''
        images = Get-Item -LiteralPath $FixtureImage
    }
    $generation = Invoke-RestMethod -Method Post -Uri "$baseUrl/api/inference/generate" -Headers $headers -WebSession $webSession -Form $form -TimeoutSec 90
    $generationJob = Wait-JobTerminal -JobId ([string]$generation.job_id) -TimeoutSeconds $InferenceTimeoutSeconds -JobLabel 'Inference'
    if ($generationJob.status -ne 'completed') { throw "Packaged inference did not complete: $($generationJob | ConvertTo-Json -Compress -Depth 8)" }

    $jobResult = $generationJob.result
    $provenance = $jobResult.provenance
    if ($null -eq $provenance -and $null -ne $jobResult.generation) { $provenance = $jobResult.generation.provenance }
    if ($null -eq $provenance -and $null -ne $jobResult.metadata) { $provenance = $jobResult.metadata.provenance }
    if ($null -eq $provenance) { throw "Packaged inference result did not include runtime provenance: $($jobResult | ConvertTo-Json -Compress -Depth 8)" }
    $runtimeProvenance = $provenance.runtime
    if ($null -eq $runtimeProvenance) { $runtimeProvenance = $provenance }

    $encodedModelRef = [Uri]::EscapeDataString($ModelRef)
    $history = Invoke-JsonApi -Method Get -Uri "$baseUrl/api/inference/history?model_ref=$encodedModelRef&limit=5"
    $historyItem = @($history.items | Where-Object { $_.model_ref -eq $ModelRef } | Select-Object -First 1)[0]
    if ($null -eq $historyItem) { throw 'Successful packaged inference was not persisted to local history.' }
    if ($historyItem.model_revision -ne $ModelRevision) { throw "History used revision '$($historyItem.model_revision)', expected '$ModelRevision'." }

    if ($Variant -eq 'cuda') {
        if (-not $runtimeProvenance.cuda_available -or -not $runtimeProvenance.cuda_used -or $runtimeProvenance.resolved_device -ne 'cuda:0') {
            throw "CUDA package did not prove GPU execution on cuda:0: $($runtimeProvenance | ConvertTo-Json -Compress -Depth 6)"
        }
    } elseif ($runtimeProvenance.cuda_used -or $runtimeProvenance.resolved_device -notmatch '^cpu(?::0)?$') {
        throw "CPU package did not prove CPU execution: $($runtimeProvenance | ConvertTo-Json -Compress -Depth 6)"
    }

    $receipt.inference = [ordered]@{
        job_id = [string]$generation.job_id
        status = [string]$generationJob.status
        history_request_id = [string]$historyItem.request_id
        history_status = [string]$historyItem.status
        model_revision = [string]$historyItem.model_revision
        generation_profile = [string]$historyItem.generation_profile
        report_count = [int]$historyItem.report_count
        image_names = @($historyItem.image_names)
        provenance = $provenance
        runtime_provenance = $runtimeProvenance
    }
    if ($historyItem.status -ne 'succeeded' -or [int]$historyItem.report_count -lt 1) { throw 'Inference history did not record a successful report.' }
    $receipt.passed = $true
} catch {
    $receipt.error = $_.Exception.Message
} finally {
    $forcedKill = $false
    if (-not $windowProcess -and $launcher -and -not $launcher.HasExited) {
        try { $windowProcess = Find-PackagedWindowProcess -LauncherProcessId $launcher.Id -TimeoutSeconds 5 } catch { }
    }
    if ($windowProcess -and -not $windowProcess.HasExited) {
        try {
            [void]$windowProcess.CloseMainWindow()
            if (-not $windowProcess.WaitForExit(30000)) {
                Stop-Process -Id $windowProcess.Id -Force -ErrorAction SilentlyContinue
                $forcedKill = $true
            }
            $windowProcess.Refresh()
        } catch { }
    }
    if ($launcher -and -not $launcher.HasExited) {
        try {
            [void]$launcher.CloseMainWindow()
            if (-not $launcher.WaitForExit(30000)) {
                Stop-Process -Id $launcher.Id -Force -ErrorAction SilentlyContinue
                $forcedKill = $true
            }
            $launcher.Refresh()
        } catch { }
    }
    if ($launcher) {
        $backendAndListenerGone = $null
        if ($backendPid -gt 0 -and $port -gt 0) {
            $backendAndListenerGone = Wait-BackendGone -ProcessId $backendPid -Port $port -TimeoutSeconds 45
        }
        $windowExited = if ($windowProcess) { $windowProcess.HasExited } elseif ($launcher) { $launcher.HasExited } else { $true }
        $receipt.cleanup = [ordered]@{
            window_exited = [bool]$windowExited
            launcher_exited = [bool]$launcher.HasExited
            backend_and_listener_gone = $backendAndListenerGone
            backend_pid = if ($backendPid -gt 0) { $backendPid } else { $null }
            port = if ($port -gt 0) { $port } else { $null }
            forced_kill = $forcedKill
        }
        if (-not $receipt.cleanup.window_exited -or -not $receipt.cleanup.launcher_exited -or
            $forcedKill -or ($null -ne $backendAndListenerGone -and -not $backendAndListenerGone)) {
            $receipt.passed = $false
            if (-not $receipt.error) { $receipt.error = 'Packaged inference process or listener cleanup did not complete gracefully.' }
        }
    }
    foreach ($contractPath in @($sessionPath, $readyPath)) {
        if (Test-Path -LiteralPath $contractPath -PathType Leaf) { Remove-Item -LiteralPath $contractPath -Force -ErrorAction SilentlyContinue }
    }
    if ($null -eq $previousLocalAppData) { Remove-Item Env:LOCALAPPDATA -ErrorAction SilentlyContinue } else { $env:LOCALAPPDATA = $previousLocalAppData }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Output) -Force | Out-Null
    $receipt.observed_utc = [DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $Output -Encoding utf8
}

if (-not $receipt.passed) { throw "Packaged inference validation failed: $($receipt | ConvertTo-Json -Compress -Depth 10)" }
Write-Host "Packaged inference validation passed: $Output"
