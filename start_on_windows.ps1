[CmdletBinding()]
param(
    [ValidateSet('Launch', 'Install', 'InitializeDatabase', 'Test', 'RemoveLogs', 'ClearCache', 'Uninstall')]
    [string]$Action,
    [switch]$Launch
)

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
$NodeVersion = '22.22.3'
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
                return $true
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    } while ((Get-Date) -lt $deadline)
    throw "Health check timed out after $TimeoutSeconds seconds: $Uri"
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

    $portableNodeNeedsUpgrade = $false
    if (Test-Path -LiteralPath $NodeExe) {
        $existingNodeVersion = (& $NodeExe --version).TrimStart('v')
        try {
            $portableNodeNeedsUpgrade = ([version]$existingNodeVersion -lt [version]$NodeVersion)
        } catch {
            $portableNodeNeedsUpgrade = $true
        }
    }
    if (-not (Test-Path -LiteralPath $NodeExe) -or $portableNodeNeedsUpgrade) {
        if ($portableNodeNeedsUpgrade) {
            Write-Info "Upgrading portable Node.js from $existingNodeVersion to $NodeVersion"
            Get-ChildItem -LiteralPath $NodeDir -Force | Remove-Item -Recurse -Force
        }
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
        FASTAPI_PORT = '5003'
        UI_HOST = '127.0.0.1'
        UI_PORT = '8003'
        UI_API_BASE_URL = '/api'
        RELOAD = 'false'
        BACKEND_VISIBLE = 'false'
        ALWAYS_REBUILD = 'false'
    }

    $environmentSource = $EnvFile
    if (-not (Test-Path -LiteralPath $environmentSource)) {
        if (-not (Test-Path -LiteralPath $EnvExample)) {
            throw "Missing environment template: $EnvExample"
        }
        $environmentSource = $EnvExample
    }

    foreach ($line in Get-Content -LiteralPath $environmentSource) {
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
    param(
        [hashtable]$Settings,
        [switch]$BuildFrontend,
        [ValidateSet('Standard', 'Development')]
        [string]$InstallationType = 'Standard'
    )

    Write-Step 'Synchronizing Python dependencies'
    $syncArgs = @('sync', '--python', $PythonExe)
    if ($InstallationType -eq 'Development') { $syncArgs += '--all-extras' }
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

    if ($BuildFrontend) {
        Write-Step 'Building frontend'
        Invoke-Checked -FilePath $NpmCmd -ArgumentList @('run', 'build') -WorkingDirectory $ClientDir
    }
}

function Test-DependenciesReady {
    $frontendPackage = Join-Path $ClientDir 'package.json'
    $frontendLock = Join-Path $ClientDir 'package-lock.json'
    $frontendModules = Join-Path $ClientDir 'node_modules'
    $frontendInstallState = Join-Path $frontendModules '.package-lock.json'
    $frontendRunner = Join-Path $frontendModules '.bin\ng.cmd'
    $backendEntrypoint = Join-Path $ServerDir 'app.py'

    if (-not (Test-Path -LiteralPath $PythonExe) -or
        -not (Test-Path -LiteralPath $UvExe) -or
        -not (Test-Path -LiteralPath $NodeExe) -or
        -not (Test-Path -LiteralPath $NpmCmd) -or
        -not (Test-Path -LiteralPath $VenvPython) -or
        -not (Test-Path -LiteralPath $backendEntrypoint) -or
        -not (Test-Path -LiteralPath $frontendPackage) -or
        -not (Test-Path -LiteralPath $frontendLock) -or
        -not (Test-Path -LiteralPath $frontendInstallState) -or
        -not (Test-Path -LiteralPath $frontendRunner)) {
        return $false
    }

    & $PythonExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $UvExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $NodeExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $VenvPython -c 'import fastapi, uvicorn' *> $null
    if ($LASTEXITCODE -ne 0) { return $false }

    return $true
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
    $settings = Import-XReportEnvironment
    Initialize-Environment
    $frontendBuilt = $false
    if (-not (Test-DependenciesReady)) {
        Write-Step 'Required application environments are missing or unusable; installing dependencies.'
        Ensure-PortableRuntimes
        Install-Dependencies -Settings $settings -BuildFrontend:($settings.ALWAYS_REBUILD -eq 'true') -InstallationType 'Standard'
        $frontendBuilt = $settings.ALWAYS_REBUILD -eq 'true'
    }
    else {
        Write-Ok 'Application environments are ready; skipped dependency installation.'
    }

    if (-not $frontendBuilt -and $settings.ALWAYS_REBUILD -eq 'true') {
        Write-Step 'Rebuilding frontend.'
        Invoke-Checked -FilePath $NpmCmd -ArgumentList @('run', 'build') -WorkingDirectory $ClientDir
    }

    Stop-PortListener -Port ([int]$settings.FASTAPI_PORT)
    Stop-PortListener -Port ([int]$settings.UI_PORT)

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Virtual-environment Python was not found at $VenvPython."
    }
    $backendAppPath = Join-Path $RepoRoot 'app'
    $backendArgs = "-m uvicorn server.app:app --app-dir `"$backendAppPath`" --host $($settings.FASTAPI_HOST) --port $($settings.FASTAPI_PORT) --log-level info"
    if ($settings.RELOAD -eq 'true') { $backendArgs += ' --reload' }

    Write-Step 'Starting backend'
    if ($settings.BACKEND_VISIBLE -eq 'true') {
        $escapedPython = $VenvPython.Replace("'", "''")
        $escapedApp = $backendAppPath.Replace("'", "''")
        $backendCommand = "& '$escapedPython' -m uvicorn server.app:app --app-dir '$escapedApp' --host $($settings.FASTAPI_HOST) --port $($settings.FASTAPI_PORT) --log-level info"
        if ($settings.RELOAD -eq 'true') { $backendCommand += ' --reload' }
        $backendProcess = Start-Process -FilePath 'powershell.exe' `
            -ArgumentList @('-NoProfile', '-NoExit', '-Command', $backendCommand) `
            -WorkingDirectory $RepoRoot -WindowStyle Normal -PassThru
    }
    else {
        $backendProcess = Start-Process -FilePath $VenvPython `
            -ArgumentList $backendArgs -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru
    }

    $healthUrl = "http://$($settings.FASTAPI_HOST):$($settings.FASTAPI_PORT)/api/health"
    Write-Step "Waiting for backend health at $healthUrl"
    Invoke-HealthCheck -Uri $healthUrl -TimeoutSeconds 60

    Write-Step 'Starting frontend preview'
    $uiUrl = "http://$($settings.UI_HOST):$($settings.UI_PORT)"
    $frontendProcess = Start-Process -FilePath $NpmCmd -ArgumentList @(
        'run', 'preview', '--', '--host', $settings.UI_HOST, '--port', $settings.UI_PORT
    ) `
        -WorkingDirectory $ClientDir -WindowStyle Hidden -PassThru
    try {
        Write-Step "Waiting for frontend at $uiUrl"
        Invoke-HealthCheck -Uri "$uiUrl/" -TimeoutSeconds 60
    }
    catch {
        if ($frontendProcess -and -not $frontendProcess.HasExited) {
            & taskkill.exe /PID $frontendProcess.Id /T /F | Out-Null
        }
        throw
    }

    Start-Process $uiUrl

    $backendPid = Get-PortProcessId -Port ([int]$settings.FASTAPI_PORT)
    Write-Ok 'Application started successfully'
    Write-Host "Backend: $healthUrl (PID $backendPid)"
    Write-Host "Frontend: $uiUrl (PID $($frontendProcess.Id))"
}

function Invoke-InstallOrUpdate {
    Ensure-PortableRuntimes
    Write-Ok 'Portable runtimes ready.'
    $installationType = Read-InstallationType
    $settings = Import-XReportEnvironment
    Stop-PortListener -Port ([int]$settings.UI_PORT)
    Install-Dependencies -Settings $settings -BuildFrontend -InstallationType $installationType
    Write-Step 'Pruning uv cache'
    Remove-Item -LiteralPath $UvCacheDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Ok 'Dependencies installed and frontend built successfully'
}

function Read-InstallationType {
    Write-Host '  [1] Development - include Ruff, Pyright, and pytest'
    Write-Host '  [2] Standard    - install runtime dependencies only'
    $selection = (Read-Host '  Select installation profile [1-2]').Trim()
    switch ($selection) {
        '1' { return 'Development' }
        '2' { return 'Standard' }
        default { throw 'Invalid installation profile. Enter 1 for Development or 2 for Standard.' }
    }
}

function Invoke-InitializeDatabase {
    Ensure-PortableRuntimes
    Initialize-Environment
    if (-not (Test-Path -LiteralPath $InitDatabaseScript)) { throw "Missing database script: $InitDatabaseScript" }
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = Join-Path $RepoRoot 'app'
    try {
        Invoke-Checked -FilePath $UvExe -ArgumentList @(
            'run', '--project', 'app/server', '--python', $PythonExe, 'python',
            'app/scripts/initialize_database.py'
        ) -WorkingDirectory $RepoRoot
    } finally {
        if ($null -eq $previousPythonPath) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $previousPythonPath
        }
    }
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

function Write-MenuRule {
    param([ConsoleColor]$Color = [ConsoleColor]::DarkCyan)
    Write-Host ('  ' + ('-' * 68)) -ForegroundColor $Color
}

function Write-MenuItem {
    param(
        [Parameter(Mandatory = $true)][string]$Number,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Description,
        [ConsoleColor]$NumberColor = [ConsoleColor]::Cyan
    )

    Write-Host ("  [{0}] " -f $Number) -NoNewline -ForegroundColor $NumberColor
    Write-Host $Label.PadRight(31) -NoNewline -ForegroundColor White
    Write-Host $Description -ForegroundColor DarkGray
}

function Show-Menu {
    Clear-Host
    Write-Host ''
    Write-Host '  XREPORT' -ForegroundColor Cyan -NoNewline
    Write-Host '  /  RADIOLOGICAL REPORTS' -ForegroundColor White
    Write-Host '  Local workspace console' -ForegroundColor DarkGray
    Write-MenuRule

    Write-Host '  APPLICATION' -ForegroundColor DarkCyan
    Write-MenuItem -Number '1' -Label 'Launch application' -Description 'Start local services'
    Write-MenuItem -Number '2' -Label 'Install / update dependencies' -Description 'Sync runtimes + packages'

    Write-Host ''
    Write-Host '  DATA & QUALITY' -ForegroundColor DarkCyan
    Write-MenuItem -Number '3' -Label 'Initialize database' -Description 'Prepare local data store'
    Write-MenuItem -Number '4' -Label 'Run test suite' -Description 'Execute project checks'

    Write-Host ''
    Write-Host '  MAINTENANCE' -ForegroundColor DarkCyan
    Write-MenuItem -Number '5' -Label 'Remove logs' -Description 'Delete application logs'
    Write-MenuItem -Number '6' -Label 'Clear cache' -Description 'Remove temporary caches'
    Write-MenuItem -Number '7' -Label 'Uninstall application' -Description 'Remove generated files' -NumberColor Yellow

    Write-Host ''
    Write-MenuRule -Color DarkGray
    Write-MenuItem -Number '8' -Label 'Exit' -Description 'Close launcher' -NumberColor DarkGray
    Write-Host ''
}

if ($Launch -and $Action) {
    throw 'Use either -Launch or -Action, not both.'
}

if ($Launch) {
    Invoke-Launch
    exit 0
}

if ($Action) {
    switch ($Action) {
        'Launch' { Invoke-Launch }
        'Install' { Invoke-InstallOrUpdate }
        'InitializeDatabase' { Invoke-InitializeDatabase }
        'Test' { Invoke-TestSuite }
        'RemoveLogs' { Remove-Logs }
        'ClearCache' { Clear-ApplicationCache }
        'Uninstall' { Uninstall-Application }
    }
    exit 0
}

while ($true) {
    Show-Menu
    $selection = (Read-Host '  Select an option (1-8)').Trim()
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
