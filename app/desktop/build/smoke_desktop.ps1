[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda')]
    [Parameter(Mandatory = $true)][string]$Variant,
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$ReleaseRoot,
    [string]$DataRoot,
    [switch]$KeepDataRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
if (-not $ReleaseRoot) { $ReleaseRoot = Join-Path $repoRoot 'release' }
$portable = Join-Path $ReleaseRoot "XREPORT-v$Version-windows-x64-$Variant-portable.exe"
if (-not (Test-Path -LiteralPath $portable -PathType Leaf)) { throw "Portable artifact is missing: $portable" }
$sourceCommit = (git -C $repoRoot rev-parse HEAD).Trim()
$ownsQaRoot = [string]::IsNullOrWhiteSpace($DataRoot)
$qaRoot = if ($ownsQaRoot) {
    Join-Path ([IO.Path]::GetTempPath()) "xreport-desktop-smoke-$PID-$Variant-$Version"
} else {
    [IO.Path]::GetFullPath($DataRoot)
}
$localAppData = Join-Path $qaRoot 'localappdata'
$stateDir = Join-Path $localAppData 'XREPORT\data\state'
$sessionFile = Join-Path $stateDir 'desktop-session.json'
$readyFile = Join-Path $stateDir 'desktop-ready.json'
$shellLog = Join-Path $localAppData 'XREPORT\data\logs\desktop-shell.log'
$timer = [Diagnostics.Stopwatch]::StartNew()
$phaseTimings = [ordered]@{}
function Mark-Phase {
    param([Parameter(Mandatory = $true)][string]$Name)
    $phaseTimings[$Name] = [int64]$timer.ElapsedMilliseconds
}
$previousLocalAppData = $env:LOCALAPPDATA
$process = $null
$backendPid = $null
$port = $null
$result = [ordered]@{
    format = 1
    application = 'XREPORT'
    version = $Version
    variant = $Variant
    source_commit = $sourceCommit
    portable = [IO.Path]::GetFileName($portable)
    started = $false
    ready_contract = $false
    health = $false
    frontend = $false
    closed = $false
    backend_process_removed = $false
    listener_removed = $false
    contracts_removed = $false
    phase_timings_ms = $phaseTimings
    shell_log = $null
    data_root_preserved = $false
    verified_utc = [DateTime]::UtcNow.ToString('o')
}

try {
    New-Item -ItemType Directory -Path $localAppData -Force | Out-Null
    $env:LOCALAPPDATA = $localAppData
    $process = Start-Process -FilePath $portable -WorkingDirectory $qaRoot -PassThru
    Mark-Phase 'process_started'
    $deadline = (Get-Date).AddSeconds(150)
    do {
        if ($process.HasExited) { throw "Portable $Variant application exited during startup with code $($process.ExitCode)." }
        if (Test-Path -LiteralPath $sessionFile) { break }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    if (-not (Test-Path -LiteralPath $sessionFile)) { throw "Portable $Variant application did not create desktop-session.json." }
    Mark-Phase 'session_written'
    $session = Get-Content -LiteralPath $sessionFile -Raw | ConvertFrom-Json
    if ($session.version -ne $Version -or $session.variant -ne $Variant) { throw 'Desktop session metadata does not match the requested build.' }
    $backendPid = [int]$session.pid
    if ($backendPid -le 0) { throw 'Desktop session metadata does not contain a valid backend process id.' }
    $bootstrap = [Uri]$session.bootstrap_url
    $port = $bootstrap.Port
    if (-not (Test-Path -LiteralPath $readyFile)) { throw 'Packaged backend did not create desktop-ready.json.' }
    $ready = Get-Content -LiteralPath $readyFile -Raw | ConvertFrom-Json
    if ($ready.host -ne '127.0.0.1' -or [int]$ready.port -ne $port -or [int]$ready.pid -ne $backendPid -or
        $ready.version -ne $Version -or $ready.variant -ne $Variant) {
        throw 'Desktop readiness metadata does not match the requested build.'
    }
    $result.ready_contract = $true
    Mark-Phase 'backend_ready_contract'
    $webSession = New-Object Microsoft.PowerShell.Commands.WebRequestSession
    $bootstrapResponse = Invoke-WebRequest -UseBasicParsing -Uri $bootstrap -WebSession $webSession -TimeoutSec 10
    if ($bootstrapResponse.StatusCode -lt 200 -or $bootstrapResponse.StatusCode -ge 300) {
        throw "Desktop bootstrap did not redirect to the Angular application (status $($bootstrapResponse.StatusCode))."
    }
    Mark-Phase 'frontend_bootstrap'
    $healthUri = "http://127.0.0.1:$port/api/health"
    $token = [Uri]::UnescapeDataString(($bootstrap.Query -replace '^\?token=', ''))
    $headers = @{ 'X-XREPORT-Desktop-Token' = $token }
    $healthResponse = Invoke-WebRequest -UseBasicParsing -Uri $healthUri -Headers $headers -TimeoutSec 10
    $health = $healthResponse.Content | ConvertFrom-Json
    if ($health.status -ne 'ok' -or $health.runtime_variant -ne $Variant -or $health.version -ne $Version) { throw 'Packaged health response does not match the requested build.' }
    $result.health = $true
    Mark-Phase 'backend_health'
    $frontendResponse = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/" -WebSession $webSession -TimeoutSec 10
    if ($frontendResponse.StatusCode -lt 200 -or $frontendResponse.StatusCode -ge 300 -or $frontendResponse.Content -notmatch '<app-root') { throw 'Packaged Angular index was not served by the backend.' }
    $result.frontend = $true
    Mark-Phase 'frontend_index'
    $result.started = $true
}
finally {
    if ($process -and -not $process.HasExited) {
        try { $process.CloseMainWindow() | Out-Null } catch { }
        try { $process.WaitForExit(15000) } catch { }
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            try { $process.WaitForExit(5000) } catch { }
        }
    }
    if ($process) { $result.closed = $process.HasExited }
    Mark-Phase 'process_closed'
    if ($backendPid) {
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            if (-not (Get-Process -Id $backendPid -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 250
        }
        $result.backend_process_removed = -not (Get-Process -Id $backendPid -ErrorAction SilentlyContinue)
    }
    if ($port) {
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            if (-not (Get-NetTCPConnection -LocalPort ([int]$port) -State Listen -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 250
        }
        $result.listener_removed = -not (Get-NetTCPConnection -LocalPort ([int]$port) -State Listen -ErrorAction SilentlyContinue)
    }
    $result.contracts_removed = -not (Test-Path -LiteralPath $sessionFile) -and -not (Test-Path -LiteralPath $readyFile)
    $logReportPath = Join-Path $repoRoot "assets\QA\desktop\smoke-$Variant-$Version-shell.log"
    if (Test-Path -LiteralPath $shellLog) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $logReportPath) -Force | Out-Null
        Copy-Item -LiteralPath $shellLog -Destination $logReportPath -Force
        $result.shell_log = [IO.Path]::GetFileName($logReportPath)
    }
    if ($ownsQaRoot -and -not $KeepDataRoot -and (Test-Path -LiteralPath $qaRoot)) {
        Remove-Item -LiteralPath $qaRoot -Recurse -Force -ErrorAction SilentlyContinue
    } elseif ($KeepDataRoot) {
        $result.data_root_preserved = $true
        Write-Host "Preserved smoke data root: $qaRoot"
    }
    if ($null -eq $previousLocalAppData) { Remove-Item Env:LOCALAPPDATA -ErrorAction SilentlyContinue } else { $env:LOCALAPPDATA = $previousLocalAppData }
    $reportPath = Join-Path $repoRoot "assets\QA\desktop\smoke-$Variant-$Version.json"
    New-Item -ItemType Directory -Path (Split-Path -Parent $reportPath) -Force | Out-Null
    $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportPath -Encoding utf8
}

if (-not $result.started -or -not $result.ready_contract -or -not $result.health -or -not $result.frontend -or -not $result.closed -or
    -not $result.backend_process_removed -or -not $result.listener_removed -or -not $result.contracts_removed) {
    throw "Desktop smoke test failed: $($result | ConvertTo-Json -Compress)"
}
Write-Host "Desktop smoke test passed: $Variant $Version"
