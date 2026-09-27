[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$version = '3.1.0'
$variant = 'cuda'
$revision = '330721b9aa5bba201a3eb88eba4dd9a6607f3e7a'
$modelRef = 'huggingface:aehrc/cxrmate-multi-tf'
$modelRelative = "data/models/huggingface/installed/aehrc__cxrmate-multi-tf/$revision"
$modelSource = Join-Path $repoRoot "app\resources\models\huggingface\installed\aehrc__cxrmate-multi-tf\$revision"
$exactModuleSource = Join-Path $repoRoot "runtimes\cache\huggingface\modules\transformers_modules\_$revision\modelling_multi.py"
$metadataSource = Join-Path $repoRoot 'app\resources\models\huggingface\metadata\aehrc__cxrmate-multi-tf.json'
$fixture = Join-Path $repoRoot 'assets\QA\inference_validation_runs\qa_pa.png'
$portable = Join-Path $repoRoot "release\XREPORT-v$version-windows-x64-$variant-portable.exe"
$dataRoot = Join-Path ([IO.Path]::GetTempPath()) 'xreport-s42-cuda-inference-20260927'
$localAppData = Join-Path $dataRoot 'localappdata'
$dataPath = Join-Path $localAppData 'XREPORT\data'
$sessionFile = Join-Path $dataPath 'state\desktop-session.json'
$readyFile = Join-Path $dataPath 'state\desktop-ready.json'
$metadataDestination = Join-Path $dataPath 'models\huggingface\metadata\aehrc__cxrmate-multi-tf.json'
$modelDestination = Join-Path $dataPath "models\huggingface\installed\aehrc__cxrmate-multi-tf\$revision"
$receiptPath = Join-Path $PSScriptRoot 'packaged-cuda-inference-receipt.json'
$stdoutPath = Join-Path $PSScriptRoot 'packaged-cuda-stdout.log'
$stderrPath = Join-Path $PSScriptRoot 'packaged-cuda-stderr.log'
$process = $null
$client = $null
$fileStream = $null
$backendPid = $null
$port = $null
$previousLocalAppData = $env:LOCALAPPDATA
$timer = [Diagnostics.Stopwatch]::StartNew()
$phaseTimings = [ordered]@{}
$receipt = [ordered]@{
    format = 1
    slice = 'S42'
    application = 'XREPORT'
    version = $version
    variant = $variant
    source_commit = ((git -C $repoRoot rev-parse HEAD).Trim())
    dirty_tree = $false
    model_ref = $modelRef
    model_revision = $revision
    exact_module_sha256 = $null
    fixture = [IO.Path]::GetFileName($fixture)
    fixture_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $fixture).Hash.ToLowerInvariant()
    result = 'UNTESTED'
    error = $null
    gpu = @()
    phase_timings_ms = $phaseTimings
    catalog = $null
    settings = $null
    health = $null
    job = $null
    persisted_history = $null
    cleanup = [ordered]@{
        process_closed = $false
        backend_process_removed = $false
        listener_removed = $false
        contracts_removed = $false
        data_root_removed = $false
    }
}

function Mark-Phase {
    param([Parameter(Mandatory = $true)][string]$Name)
    $phaseTimings[$Name] = [int64]$timer.ElapsedMilliseconds
}

function Get-GpuSnapshot {
    $lines = @(& nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader 2>$null)
    if ($LASTEXITCODE -ne 0) { return @() }
    return @($lines | ForEach-Object { [string]$_ })
}

function Get-Json {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][hashtable]$Headers,
        [Parameter(Mandatory = $true)][Microsoft.PowerShell.Commands.WebRequestSession]$WebSession
    )
    $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -Headers $Headers -WebSession $WebSession -TimeoutSec 30
    return ($response.Content | ConvertFrom-Json)
}

function Stop-PackagedProcess {
    if ($null -ne $process -and -not $process.HasExited) {
        try { $process.CloseMainWindow() | Out-Null } catch { }
        try { $process.WaitForExit(15000) } catch { }
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            try { $process.WaitForExit(5000) } catch { }
        }
    }
    if ($null -ne $process) { $receipt.cleanup.process_closed = $process.HasExited }
    if ($backendPid) {
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            if (-not (Get-Process -Id $backendPid -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 250
        }
        $receipt.cleanup.backend_process_removed = -not (Get-Process -Id $backendPid -ErrorAction SilentlyContinue)
    }
    if ($port) {
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            if (-not (Get-NetTCPConnection -LocalPort ([int]$port) -State Listen -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 250
        }
        $receipt.cleanup.listener_removed = -not (Get-NetTCPConnection -LocalPort ([int]$port) -State Listen -ErrorAction SilentlyContinue)
    }
    $receipt.cleanup.contracts_removed = -not (Test-Path -LiteralPath $sessionFile) -and -not (Test-Path -LiteralPath $readyFile)
}

try {
    foreach ($path in @($portable, $modelSource, $exactModuleSource, $metadataSource, $fixture)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Required validation input is missing: $path" }
    }
    if (Test-Path -LiteralPath $dataRoot) { throw "Refusing to overwrite existing disposable root: $dataRoot" }
    New-Item -ItemType Directory -Path (Split-Path -Parent $modelDestination), (Split-Path -Parent $metadataDestination) -Force | Out-Null
    Copy-Item -LiteralPath $modelSource -Destination $modelDestination -Recurse -Force
    $moduleDestination = Join-Path $modelDestination 'modelling_multi.py'
    Copy-Item -LiteralPath $exactModuleSource -Destination $moduleDestination -Force
    $receipt.exact_module_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $moduleDestination).Hash.ToLowerInvariant()
    $metadata = Get-Content -LiteralPath $metadataSource -Raw | ConvertFrom-Json
    $metadata.active_relative_path = $modelRelative
    $metadata | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $metadataDestination -Encoding utf8
    $receipt.gpu = @(Get-GpuSnapshot)
    Mark-Phase 'fixture_prepared'

    New-Item -ItemType Directory -Path $localAppData -Force | Out-Null
    $env:LOCALAPPDATA = $localAppData
    $process = Start-Process -FilePath $portable -WorkingDirectory $dataRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    Mark-Phase 'process_started'
    $deadline = (Get-Date).AddSeconds(180)
    do {
        if ($process.HasExited) { throw "CUDA portable process exited during startup with code $($process.ExitCode)." }
        if (Test-Path -LiteralPath $sessionFile) { break }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    if (-not (Test-Path -LiteralPath $sessionFile)) { throw 'CUDA portable process did not create desktop-session.json.' }
    $session = Get-Content -LiteralPath $sessionFile -Raw | ConvertFrom-Json
    $backendPid = [int]$session.pid
    $bootstrap = [Uri]$session.bootstrap_url
    $port = $bootstrap.Port
    $token = [Uri]::UnescapeDataString(($bootstrap.Query -replace '^\?token=', ''))
    $headers = @{ 'X-XREPORT-Desktop-Token' = $token }
    $webSession = New-Object Microsoft.PowerShell.Commands.WebRequestSession
    $bootstrapResponse = Invoke-WebRequest -UseBasicParsing -Uri $bootstrap -WebSession $webSession -TimeoutSec 30
    if ([int]$bootstrapResponse.StatusCode -lt 200 -or [int]$bootstrapResponse.StatusCode -ge 400) { throw "Packaged bootstrap did not establish a session: $($bootstrapResponse.StatusCode)" }
    if (-not @($webSession.Cookies.GetCookies("http://127.0.0.1:$port/") | Where-Object { $_.Name -eq 'xreport_session' }).Count) { throw 'Packaged bootstrap did not set xreport_session.' }
    $health = $null
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        try {
            $health = Get-Json -Uri "http://127.0.0.1:$port/api/health" -Headers $headers -WebSession $webSession
            if ($health.status -eq 'ok') { break }
        }
        catch { }
        Start-Sleep -Seconds 1
    }
    if ($null -eq $health -or $health.status -ne 'ok') { throw 'Packaged CUDA backend did not become healthy.' }
    $receipt.health = $health
    Mark-Phase 'backend_healthy'

    $receipt.settings = Get-Json -Uri "http://127.0.0.1:$port/api/settings" -Headers $headers -WebSession $webSession

    $catalog = Get-Json -Uri "http://127.0.0.1:$port/api/inference/models" -Headers $headers -WebSession $webSession
    $selected = @($catalog.models | Where-Object { $_.model_ref -eq $modelRef }) | Select-Object -First 1
    $receipt.catalog = [ordered]@{
        status = $selected.status
        installation_state = $selected.installation_state
        integrity_status = $selected.integrity_status
        model_revision = $selected.model_revision
        local_path = $selected.local_path
    }
    if ($null -eq $selected -or $selected.status -ne 'ready' -or $selected.installation_state -ne 'active' -or $selected.integrity_status -ne 'verified' -or $selected.model_revision -ne $revision) {
        throw "Packaged CUDA model catalogue was not ready: $($receipt.catalog | ConvertTo-Json -Compress)"
    }
    Mark-Phase 'model_ready'

    $client = [Net.Http.HttpClient]::new()
    $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, "http://127.0.0.1:$port/api/inference/generate")
    $request.Headers.Add('X-XREPORT-Desktop-Token', $token)
    $request.Headers.Add('Cookie', "xreport_session=$token")
    $multipart = [Net.Http.MultipartFormDataContent]::new()
    $multipart.Add([Net.Http.StringContent]::new($modelRef), 'model_ref')
    $multipart.Add([Net.Http.StringContent]::new('deterministic'), 'generation_profile')
    $multipart.Add([Net.Http.StringContent]::new(''), 'clinical_context')
    $fileStream = [IO.File]::OpenRead($fixture)
    $fileContent = [Net.Http.StreamContent]::new($fileStream)
    $fileContent.Headers.ContentType = [Net.Http.Headers.MediaTypeHeaderValue]::Parse('image/png')
    $multipart.Add($fileContent, 'images', [IO.Path]::GetFileName($fixture))
    $request.Content = $multipart
    $startResponse = $client.SendAsync($request).GetAwaiter().GetResult()
    $startBody = $startResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if ([int]$startResponse.StatusCode -ne 202) { throw "Packaged CUDA inference did not start: $startBody" }
    $start = $startBody | ConvertFrom-Json
    $jobId = [string]$start.job_id
    Mark-Phase 'inference_started'

    $job = $null
    $deadline = (Get-Date).AddSeconds(600)
    do {
        $job = Get-Json -Uri "http://127.0.0.1:$port/api/jobs/$jobId" -Headers $headers -WebSession $webSession
        if ($job.status -in @('completed', 'failed', 'cancelled')) { break }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    $receipt.job = [ordered]@{
        job_id = $job.job_id
        status = $job.status
        progress = $job.progress
        error = $job.error
        runtime = $job.result.provenance.runtime
        report_count = $job.result.count
    }
    if ($job.status -ne 'completed') { throw "Packaged CUDA inference ended with status $($job.status): $($job.error)" }
    $runtime = $job.result.provenance.runtime
    if (-not $runtime.cuda_available -or -not $runtime.cuda_used -or ([string]$runtime.resolved_device) -notmatch '^cuda(:|$)') {
        throw "Packaged inference did not prove CUDA execution: $($receipt.job | ConvertTo-Json -Compress)"
    }
    $history = Get-Json -Uri "http://127.0.0.1:$port/api/inference/history?model_ref=$([Uri]::EscapeDataString($modelRef))&limit=1&sort=newest" -Headers $headers -WebSession $webSession
    $historyItem = $history.items[0]
    $historyDetail = Get-Json -Uri "http://127.0.0.1:$port/api/inference/history/$($historyItem.request_id)" -Headers $headers -WebSession $webSession
    $receipt.persisted_history = [ordered]@{
        total = $history.total
        request_id = $historyItem.request_id
        model_ref = $historyDetail.model_ref
        status = $historyDetail.status
        provenance_available = $historyItem.provenance_available
        runtime = $historyDetail.generation_config.provenance.runtime
    }
    if ($receipt.persisted_history.status -ne 'succeeded' -or -not $receipt.persisted_history.runtime.cuda_used) { throw 'Persisted inference history did not retain CUDA provenance.' }
    Mark-Phase 'inference_completed_and_persisted'
    $receipt.gpu_after = @(Get-GpuSnapshot)
    $receipt.result = 'PASS'
}
catch {
    $receipt.result = 'FAIL'
    $receipt.error = $_.Exception.Message
    throw
}
finally {
    if ($null -ne $fileStream) { $fileStream.Dispose() }
    if ($null -ne $client) { $client.Dispose() }
    Stop-PackagedProcess
    if ($null -eq $previousLocalAppData) { Remove-Item Env:LOCALAPPDATA -ErrorAction SilentlyContinue } else { $env:LOCALAPPDATA = $previousLocalAppData }
    if (Test-Path -LiteralPath $dataRoot) { Remove-Item -LiteralPath $dataRoot -Recurse -Force -ErrorAction SilentlyContinue }
    $receipt.cleanup.data_root_removed = -not (Test-Path -LiteralPath $dataRoot)
    $receipt.closed_utc = [DateTime]::UtcNow.ToString('o')
    New-Item -ItemType Directory -Path (Split-Path -Parent $receiptPath) -Force | Out-Null
    $receipt | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $receiptPath -Encoding utf8
}

Write-Host "Packaged CUDA inference validation passed: $receiptPath"
