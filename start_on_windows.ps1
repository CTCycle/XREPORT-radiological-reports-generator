[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimesDir = Join-Path $RepoRoot 'runtimes'
$PythonDir = Join-Path $RuntimesDir 'python'
$PythonExe = Join-Path $PythonDir 'python.exe'
$PythonPth = Join-Path $PythonDir 'python314._pth'
$UvDir = Join-Path $RuntimesDir 'uv'
$UvExe = Join-Path $UvDir 'uv.exe'
$UvCacheDir = Join-Path $RuntimesDir '.uv-cache'
$NodeDir = Join-Path $RuntimesDir 'nodejs'
$NodeExe = Join-Path $NodeDir 'node.exe'
$NpmCmd = Join-Path $NodeDir 'npm.cmd'
$ServerDir = Join-Path $RepoRoot 'app\server'
$ClientDir = Join-Path $RepoRoot 'app\client'
$VenvDir = Join-Path $ServerDir '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$EnvFile = Join-Path $RepoRoot 'settings\.env'
$EnvExample = Join-Path $RepoRoot 'settings\.env.example'
$TestsBat = Join-Path $RepoRoot 'app\tests\run_tests.bat'
$InitDatabaseScript = Join-Path $RepoRoot 'app\scripts\initialize_database.py'

$PythonVersion = '3.14.2'
$PythonArchive = "python-$PythonVersion-embed-amd64.zip"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonArchive"
$UvUrlAmd64 = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
$UvUrlArm64 = 'https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-pc-windows-msvc.zip'
$NodeVersion = '22.12.0'
$NodeArchive = "node-v$NodeVersion-win-x64.zip"
$NodeUrl = "https://nodejs.org/dist/v$NodeVersion/$NodeArchive"

function Write-Step([string]$Message) { Write-Host "[STEP] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Info([string]$Message) { Write-Host "[INFO] $Message" -ForegroundColor Gray }
function Write-Warn([string]$Message) { Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Fatal([string]$Message) { Write-Host "[FATAL] $Message" -ForegroundColor Red }

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = $RepoRoot
    )

    Push-Location $WorkingDirectory
    try {
        & $FilePath @ArgumentList
        if ($LASTEXITCODE -ne 0) {
            throw "$FilePath failed with exit code $LASTEXITCODE."
        }
    } finally {
        Pop-Location
    }
}

function Initialize-Environment {
    $env:UV_CACHE_DIR = $UvCacheDir
    $env:UV_PROJECT_ENVIRONMENT = $VenvDir
    $env:UV_LINK_MODE = 'copy'
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
    $env:PATH = "$NodeDir;$($env:PATH)"
}

function Invoke-DownloadAndExtract {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )
    $ProgressPreference = 'SilentlyContinue'
    New-Item -ItemType Directory -Path (Split-Path -Parent $ArchivePath) -Force | Out-Null
    New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $ArchivePath
    try {
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $DestinationPath -Force
    } finally {
        Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-PatchPythonPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path -LiteralPath $Path) {
        (Get-Content -LiteralPath $Path) -replace '^#import site$', 'import site' |
            Set-Content -LiteralPath $Path -Encoding ascii
    }
}

function Get-PythonVersion {
    param([Parameter(Mandatory = $true)][string]$PythonExe)
    & $PythonExe -c 'import platform; print(platform.python_version())'
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Find-UvExecutable {
    param([Parameter(Mandatory = $true)][string]$SearchRoot)
    $uv = Get-ChildItem -LiteralPath $SearchRoot -Recurse -File -Filter 'uv.exe' |
        Select-Object -First 1
    if ($null -eq $uv) {
        throw "uv.exe was not found under $SearchRoot"
    }
    $uv.FullName
}

function Invoke-HealthCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [ValidateRange(1, 3600)][int]$TimeoutSeconds = 60
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                exit 0
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    } while ((Get-Date) -lt $deadline)
    Write-Error "Health check timed out after $TimeoutSeconds seconds: $Uri"
    exit 1
}

function Ensure-PortableRuntimes {
    Write-Step 'Preparing portable runtimes'
    New-Item -ItemType Directory -Path $RuntimesDir, $PythonDir, $UvDir, $NodeDir -Force | Out-Null

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        Write-Info "Downloading Python $PythonVersion"
        Invoke-DownloadAndExtract -Uri $PythonUrl -ArchivePath (Join-Path $PythonDir $PythonArchive) -DestinationPath $PythonDir
    }
    Invoke-PatchPythonPath -Path $PythonPth
    $foundVersion = Get-PythonVersion -PythonExe $PythonExe
    Write-Ok "Python ready: $foundVersion"

    if (-not (Test-Path -LiteralPath $UvExe)) {
        $uvUrl = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { $UvUrlArm64 } else { $UvUrlAmd64 }
        Write-Info 'Downloading uv'
        Invoke-DownloadAndExtract -Uri $uvUrl -ArchivePath (Join-Path $UvDir 'uv.zip') -DestinationPath $UvDir
        $foundUv = Find-UvExecutable -SearchRoot $UvDir
        if ($foundUv -ne $UvExe) {
            Copy-Item -LiteralPath $foundUv -Destination $UvExe -Force
        }
    }
    Invoke-Checked -FilePath $UvExe -ArgumentList @('--version')

    if (-not (Test-Path -LiteralPath $NodeExe)) {
        Write-Info "Downloading Node.js $NodeVersion"
        Invoke-DownloadAndExtract -Uri $NodeUrl -ArchivePath (Join-Path $NodeDir $NodeArchive) -DestinationPath $NodeDir
    }
    $nestedNodeDir = Join-Path $NodeDir "node-v$NodeVersion-win-x64"
    if (Test-Path -LiteralPath (Join-Path $nestedNodeDir 'node.exe')) {
        Get-ChildItem -LiteralPath $nestedNodeDir -Force | Move-Item -Destination $NodeDir -Force
        Remove-Item -LiteralPath $nestedNodeDir -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath $NodeExe) -or -not (Test-Path -LiteralPath $NpmCmd)) {
        throw "Portable Node.js or npm was not found in $NodeDir."
    }
    $nodeVersionOutput = & $NodeExe --version
    Write-Ok "Node.js ready: $nodeVersionOutput"
    Initialize-Environment
}

function Import-XReportEnvironment {
    $values = @{
        FASTAPI_HOST = '127.0.0.1'
        FASTAPI_PORT = '8000'
        UI_HOST = '127.0.0.1'
        UI_PORT = '8001'
        RELOAD = 'false'
        OPTIONAL_DEPENDENCIES = 'false'
        BACKEND_VISIBLE = 'false'
    }

    if (-not (Test-Path -LiteralPath $EnvFile)) {
        if (-not (Test-Path -LiteralPath $EnvExample)) {
            throw "Missing environment template: $EnvExample"
        }
        Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
        Write-Info "Created first-launch settings file: $EnvFile"
    }

    foreach ($line in Get-Content -LiteralPath $EnvFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or $trimmed.StartsWith(';') -or -not $trimmed.Contains('=')) {
            continue
        }
        $parts = $trimmed.Split('=', 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($key) {
            $values[$key] = $value
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }
    return $values
}

function Install-Dependencies {
    param([hashtable]$Settings)

    Write-Step 'Synchronizing Python dependencies'
    $syncArgs = @('sync', '--python', $PythonExe)
    if ($Settings.OPTIONAL_DEPENDENCIES -eq 'true') { $syncArgs += '--all-extras' }
    try {
        Invoke-Checked -FilePath $UvExe -ArgumentList $syncArgs -WorkingDirectory $ServerDir
    } catch {
        Write-Warn 'Recreating the project virtual environment after a failed sync'
        Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
        Invoke-Checked -FilePath $UvExe -ArgumentList $syncArgs -WorkingDirectory $ServerDir
    }

    Write-Step 'Installing frontend dependencies'
    $npmInstallArgs = if (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json')) { @('ci') } else { @('install') }
    Invoke-Checked -FilePath $NpmCmd -ArgumentList $npmInstallArgs -WorkingDirectory $ClientDir

    Write-Step 'Building frontend'
    Invoke-Checked -FilePath $NpmCmd -ArgumentList @('run', 'build') -WorkingDirectory $ClientDir
}

function Stop-PortListener {
    param([Parameter(Mandatory = $true)][int]$Port)

    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $listeners) {
        Write-Info "Releasing port $Port from PID $processId"
        & taskkill.exe /PID $processId /T /F | Out-Null
    }
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) { return }
        Start-Sleep -Seconds 1
    }
    throw "Port $Port is still occupied after 20 seconds."
}

function Get-PortProcessId {
    param([int]$Port)
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty OwningProcess
}

function Invoke-Launch {
    Ensure-PortableRuntimes
    $settings = Import-XReportEnvironment
    Install-Dependencies -Settings $settings
    Stop-PortListener -Port ([int]$settings.FASTAPI_PORT)
    Stop-PortListener -Port ([int]$settings.UI_PORT)

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Virtual-environment Python was not found at $VenvPython."
    }
    $reloadArgs = if ($settings.RELOAD -eq 'true') { ' --reload' } else { '' }
    $backendArgs = "-m uvicorn server.app:app --app-dir `"$(Join-Path $RepoRoot 'app')`" --host $($settings.FASTAPI_HOST) --port $($settings.FASTAPI_PORT)$reloadArgs --log-level info"

    Write-Step 'Starting backend'
    if ($settings.BACKEND_VISIBLE -eq 'true') {
        $visibleCommand = "start `"Backend`" cmd /c `"`"$VenvPython`" $backendArgs`""
        Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $visibleCommand -Wait
    } else {
        Start-Process -FilePath $VenvPython -ArgumentList $backendArgs -WorkingDirectory $RepoRoot -WindowStyle Hidden | Out-Null
    }

    $healthUrl = "http://$($settings.FASTAPI_HOST):$($settings.FASTAPI_PORT)/api/health"
    Write-Step "Waiting for backend health at $healthUrl"
    Invoke-HealthCheck -Uri $healthUrl -TimeoutSeconds 60
    if ($LASTEXITCODE -ne 0) { throw 'Backend health check failed.' }

    Write-Step 'Starting frontend preview'
    $frontendProcess = Start-Process -FilePath $NpmCmd -ArgumentList @(
        'run', 'preview', '--', '--host', $settings.UI_HOST, '--port', $settings.UI_PORT, '--strictPort'
    ) -WorkingDirectory $ClientDir -WindowStyle Hidden -PassThru
    $uiUrl = "http://$($settings.UI_HOST):$($settings.UI_PORT)"
    Start-Process $uiUrl

    $backendPid = Get-PortProcessId -Port ([int]$settings.FASTAPI_PORT)
    Write-Ok 'Application started successfully'
    Write-Host "Backend: $healthUrl (PID $backendPid)"
    Write-Host "Frontend: $uiUrl (PID $($frontendProcess.Id))"
}

function Invoke-InstallOrUpdate {
    Ensure-PortableRuntimes
    $settings = Import-XReportEnvironment
    Install-Dependencies -Settings $settings
    Write-Step 'Pruning uv cache'
    Remove-Item -LiteralPath $UvCacheDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Ok 'Dependencies installed and frontend built successfully'
}

function Invoke-InitializeDatabase {
    Ensure-PortableRuntimes
    Initialize-Environment
    if (-not (Test-Path -LiteralPath $InitDatabaseScript)) { throw "Missing database script: $InitDatabaseScript" }
    Invoke-Checked -FilePath $UvExe -ArgumentList @(
        'run', '--project', 'app/server', '--python', $PythonExe, 'python',
        'app/scripts/initialize_database.py', '--drop-existing', '--seed-catalogs', '--force-reseed-catalogs'
    ) -WorkingDirectory $RepoRoot
    Write-Ok 'Database initialization completed'
}

function Invoke-TestSuite {
    if (-not (Test-Path -LiteralPath $TestsBat)) { throw "Missing test script: $TestsBat" }
    Write-Step "Executing test suite: $TestsBat"
    & $TestsBat
    $testExitCode = $LASTEXITCODE
    if ($testExitCode -ne 0) { throw "Test suite failed with exit code $testExitCode." }
    Write-Ok 'Test suite completed successfully'
}

function Remove-Logs {
    $logDir = Join-Path $RepoRoot 'app\resources\logs'
    $logs = Get-ChildItem -LiteralPath $logDir -File -Filter '*.log' -ErrorAction SilentlyContinue
    if ($logs) {
        $logs | Remove-Item -Force
        Write-Ok "Removed $($logs.Count) log file(s)"
    } else {
        Write-Info 'No log files found'
    }
}

function Remove-PythonCaches {
    $caches = Get-ChildItem -LiteralPath $RepoRoot -Directory -Recurse -Filter '__pycache__' -Force -ErrorAction SilentlyContinue
    foreach ($cache in $caches) { Remove-Item -LiteralPath $cache.FullName -Recurse -Force }
    Write-Ok "Removed $($caches.Count) Python cache directorie(s)"
}

function Clear-ApplicationCache {
    Remove-PythonCaches
    Remove-Item -LiteralPath $UvCacheDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Ok 'Application caches cleared'
}

function Uninstall-Application {
    $targets = @(
        $RuntimesDir,
        $VenvDir,
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $ClientDir 'node_modules'),
        (Join-Path $ClientDir '.angular'),
        (Join-Path $ClientDir 'dist'),
        (Join-Path $ClientDir 'package-lock.json'),
        (Join-Path $ServerDir 'uv.lock'),
        (Join-Path $RepoRoot 'uv.lock')
    )
    foreach ($target in $targets) {
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    }
    Remove-PythonCaches
    Write-Ok 'Application runtimes and generated dependencies removed; settings and user data were preserved'
}

function Show-Menu {
    Clear-Host
    Write-Host '========================================='
    Write-Host '    XREPORT -- Radiological Reports Generator'
    Write-Host '========================================='
    Write-Host '1.  Launch application'
    Write-Host '2.  Install / update dependencies'
    Write-Host '3.  Initialize database'
    Write-Host '4.  Run test suite'
    Write-Host '5.  Remove logs'
    Write-Host '6.  Clear cache'
    Write-Host '7.  Uninstall application'
    Write-Host '8.  Exit'
    Write-Host '========================================='
}

while ($true) {
    Show-Menu
    $selection = (Read-Host 'Select an option (1-8)').Trim()
    if ($selection -notmatch '^[1-8]$') {
        Write-Warn 'Invalid option. Enter a number from 1 to 8.'
        [void](Read-Host 'Press Enter to continue')
        continue
    }
    if ($selection -eq '8') { break }

    try {
        switch ($selection) {
            '1' { Invoke-Launch; exit 0 }
            '2' { Invoke-InstallOrUpdate }
            '3' { Invoke-InitializeDatabase }
            '4' { Invoke-TestSuite }
            '5' { Remove-Logs }
            '6' { Clear-ApplicationCache }
            '7' { Uninstall-Application }
        }
    } catch {
        Write-Fatal $_.Exception.Message
    }
    Write-Host 'Press any key to return to menu...'
    [void]$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
}
