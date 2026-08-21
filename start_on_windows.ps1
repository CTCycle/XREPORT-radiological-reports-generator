[CmdletBinding()]
param(
    [ValidateSet('Launch', 'LaunchDesktopDev', 'BuildDesktopRelease', 'RemoveDesktopRelease', 'Install', 'RebuildFrontend', 'InitializeDatabase', 'Test', 'RemoveLogs', 'ClearCache', 'Uninstall')]
    [string]$Action,
    [switch]$Launch,
    [ValidateSet('Cpu', 'Cuda', 'All')]
    [string]$DesktopRuntime = 'All',
    [ValidateSet('Portable', 'Msi', 'All')]
    [string]$DesktopTarget = 'All',
    [switch]$OfflineWebView2,
    [string]$Version = '1.0.0',
    [switch]$Force,
    [switch]$AllowDirtyTree
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimesDir = Join-Path $RepoRoot 'runtimes'
$PythonDir = Join-Path $RuntimesDir 'python'
$PythonExe = Join-Path $PythonDir 'python.exe'
$PythonPth = Join-Path $PythonDir 'python314._pth'
$UvDir = Join-Path $RuntimesDir 'uv'
$UvExe = Join-Path $UvDir 'uv.exe'
$RuntimeCacheDir = Join-Path $RuntimesDir 'cache'
$UvCacheDir = Join-Path $RuntimeCacheDir 'uv'
$NpmCacheDir = Join-Path $RuntimeCacheDir 'npm'
$PipCacheDir = Join-Path $RuntimeCacheDir 'pip'
$PlaywrightBrowsersCacheDir = Join-Path $RuntimeCacheDir 'playwright-browsers'
$ToolCacheDir = Join-Path $RepoRoot 'app\tests\cache'
$PytestCacheDir = Join-Path $ToolCacheDir 'pytest'
$PytestTempDir = Join-Path $ToolCacheDir 'pytest-tmp'
$RuffCacheDir = Join-Path $ToolCacheDir 'ruff'
$MypyCacheDir = Join-Path $ToolCacheDir 'mypy'
$PythonCacheDir = Join-Path $ToolCacheDir 'python'
$CoverageCacheDir = Join-Path $ToolCacheDir 'coverage'
$AngularCacheDir = Join-Path $ToolCacheDir 'angular'
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
$DesktopDir = Join-Path $RepoRoot 'app\desktop'
$DesktopTauriDir = Join-Path $DesktopDir 'src-tauri'
$DesktopBuildDir = Join-Path $DesktopDir 'build'
$DesktopReleaseDir = Join-Path $RepoRoot 'release'
$DesktopTargetDir = Join-Path $DesktopTauriDir 'target'
$DesktopPythonScript = Join-Path $DesktopBuildDir 'run_pyinstaller.py'
$DesktopSpec = Join-Path $DesktopBuildDir 'xreport_backend.spec'
$DesktopBundleScript = Join-Path $DesktopBuildDir 'create_runtime_bundle.py'

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
    New-Item -ItemType Directory -Path @(
        $RuntimeCacheDir, $UvCacheDir, $NpmCacheDir, $PipCacheDir,
        $PlaywrightBrowsersCacheDir, $ToolCacheDir, $PytestCacheDir,
        $PytestTempDir, $RuffCacheDir, $MypyCacheDir, $PythonCacheDir,
        $CoverageCacheDir, $AngularCacheDir
    ) -Force | Out-Null
    $env:UV_CACHE_DIR = $UvCacheDir
    $env:PIP_CACHE_DIR = $PipCacheDir
    $env:NPM_CONFIG_CACHE = $NpmCacheDir
    $env:npm_config_cache = $NpmCacheDir
    $env:PLAYWRIGHT_BROWSERS_PATH = $PlaywrightBrowsersCacheDir
    $env:XDG_CACHE_HOME = $ToolCacheDir
    $env:RUFF_CACHE_DIR = $RuffCacheDir
    $env:MYPY_CACHE_DIR = $MypyCacheDir
    $env:PYTHONPYCACHEPREFIX = $PythonCacheDir
    $env:COVERAGE_FILE = Join-Path $CoverageCacheDir '.coverage'
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

    Ensure-PortableNodeRuntime
    Initialize-Environment
}

function Ensure-PortableNodeRuntime {
    New-Item -ItemType Directory -Path $NodeDir -Force | Out-Null
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

    Install-FrontendDependencies

    if ($BuildFrontend) {
        Invoke-FrontendBuild
    }
}

function Install-FrontendDependencies {
    Write-Step 'Installing frontend dependencies'
    $npmInstallArgs = if (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json')) { @('ci') } else { @('install') }
    Invoke-Checked -FilePath $NpmCmd -ArgumentList $npmInstallArgs -WorkingDirectory $ClientDir
}

function Invoke-FrontendBuild {
    Write-Step 'Building frontend'
    Invoke-Checked -FilePath $NpmCmd -ArgumentList @('run', 'build') -WorkingDirectory $ClientDir
}

function Test-FrontendDependenciesReady {
    $frontendPackage = Join-Path $ClientDir 'package.json'
    $frontendLock = Join-Path $ClientDir 'package-lock.json'
    $frontendModules = Join-Path $ClientDir 'node_modules'
    $frontendInstallState = Join-Path $frontendModules '.package-lock.json'
    $frontendRunner = Join-Path $frontendModules '.bin\ng.cmd'

    return (Test-Path -LiteralPath $NodeExe) -and
        (Test-Path -LiteralPath $NpmCmd) -and
        (Test-Path -LiteralPath $frontendPackage) -and
        (Test-Path -LiteralPath $frontendLock) -and
        (Test-Path -LiteralPath $frontendInstallState) -and
        (Test-Path -LiteralPath $frontendRunner)
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
        Invoke-FrontendBuild
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
    Write-Step 'Synchronizing database schema'
    Invoke-InitializeDatabase
    Write-Ok 'Dependencies installed and frontend built successfully'
}

function Invoke-RebuildFrontend {
    $settings = Import-XReportEnvironment
    Ensure-PortableNodeRuntime
    Initialize-Environment
    if (-not (Test-FrontendDependenciesReady)) {
        Write-Step 'Frontend dependencies are missing or unusable; installing them.'
        Install-FrontendDependencies
    }
    Stop-PortListener -Port ([int]$settings.UI_PORT)
    Invoke-FrontendBuild
    Write-Ok 'Frontend rebuilt successfully'
}

function Get-DesktopVariants {
    param([string]$Runtime = $DesktopRuntime)
    switch ($Runtime) {
        'Cpu' { @('cpu') }
        'Cuda' { @('cuda') }
        default { @('cpu', 'cuda') }
    }
}

function Get-DesktopVersionMetadata {
    $clientVersion = ([string]((Get-Content -LiteralPath (Join-Path $ClientDir 'package.json') -Raw | ConvertFrom-Json).version)).Trim()
    $serverVersion = (Select-String -LiteralPath (Join-Path $ServerDir 'pyproject.toml') -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches.Groups[1].Value
    $backendVersion = (Select-String -LiteralPath (Join-Path $ServerDir 'common\constants.py') -Pattern 'FASTAPI_VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches.Groups[1].Value
    $cargoVersion = (Select-String -LiteralPath (Join-Path $DesktopTauriDir 'Cargo.toml') -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches.Groups[1].Value
    $tauriVersions = @()
    foreach ($config in @('tauri.cpu.conf.json', 'tauri.cuda.conf.json')) {
        $configPath = Join-Path $DesktopTauriDir $config
        $tauriVersions += ([string]((Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).version)).Trim()
    }
    [pscustomobject]@{
        Client = $clientVersion
        Server = $serverVersion
        Backend = $backendVersion
        Cargo = $cargoVersion
        Tauri = ($tauriVersions -join ',')
    }
}

function Assert-DesktopVersion {
    param([Parameter(Mandatory = $true)][string]$ExpectedVersion)
    $metadata = Get-DesktopVersionMetadata
    $values = @($metadata.Client, $metadata.Server, $metadata.Backend, $metadata.Cargo) + @($metadata.Tauri -split ',')
    if (($values | Where-Object { $_ -ne $ExpectedVersion }).Count -gt 0) {
        throw "Desktop version drift detected. Expected $ExpectedVersion; metadata: $($metadata | ConvertTo-Json -Compress)"
    }
    Write-Ok "Desktop version metadata is synchronized at $ExpectedVersion"
}

function Assert-DesktopSourceState {
    $status = @(git -C $RepoRoot status --porcelain)
    if ($status.Count -gt 0 -and -not $AllowDirtyTree) {
        throw 'Desktop release requires a clean git tree. Use -AllowDirtyTree only for diagnostic builds.'
    }
    if ($status.Count -gt 0) {
        Write-Warn 'Building from a dirty tree because -AllowDirtyTree was supplied.'
    }
    return [pscustomobject]@{
        Dirty = ($status.Count -gt 0)
        Commit = ((git -C $RepoRoot rev-parse HEAD).Trim())
    }
}

function Get-DesktopConfigPath {
    param(
        [Parameter(Mandatory = $true)][string]$Variant,
        [string]$ReleaseVersion = $Version
    )
    $sourceName = if ($Variant -eq 'cpu') { 'tauri.cpu.conf.json' } else { 'tauri.cuda.conf.json' }
    $sourcePath = Join-Path $DesktopTauriDir $sourceName
    $configPath = Join-Path $DesktopBuildDir "tauri-$Variant-$ReleaseVersion.json"
    New-Item -ItemType Directory -Path $DesktopBuildDir -Force | Out-Null
    $config = Get-Content -LiteralPath $sourcePath -Raw | ConvertFrom-Json
    if ($OfflineWebView2) {
        $config.bundle.windows.webviewInstallMode = [pscustomobject]@{ type = 'offlineInstaller' }
    }
    $config | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $configPath -Encoding utf8
    return $configPath
}

function Invoke-DesktopFrontendBuild {
    if (-not (Test-FrontendDependenciesReady)) {
        Ensure-PortableNodeRuntime
        foreach ($line in @(Install-FrontendDependencies)) { Write-Host $line }
    }
    foreach ($line in @(Invoke-FrontendBuild)) { Write-Host $line }
    $frontendOutput = Join-Path $ClientDir 'dist\client-angular\browser\index.html'
    if (-not (Test-Path -LiteralPath $frontendOutput)) {
        throw "Angular production output was not created: $frontendOutput"
    }
    return (Split-Path -Parent $frontendOutput)
}

function Invoke-DesktopBackendFreeze {
    param(
        [Parameter(Mandatory = $true)][string]$Variant,
        [Parameter(Mandatory = $true)][string]$SourceCommit,
        [Parameter(Mandatory = $true)][string]$FrontendDist
    )
    $stagingRoot = Join-Path $DesktopBuildDir "runtime-staging\$Variant"
    $distRoot = Join-Path $DesktopBuildDir "pyinstaller\$Variant\dist"
    $workRoot = Join-Path $DesktopBuildDir "pyinstaller\$Variant\work"
    if (Test-Path -LiteralPath $stagingRoot) { Remove-Item -LiteralPath $stagingRoot -Recurse -Force }
    if (Test-Path -LiteralPath $distRoot) { Remove-Item -LiteralPath $distRoot -Recurse -Force }
    if (Test-Path -LiteralPath $workRoot) { Remove-Item -LiteralPath $workRoot -Recurse -Force }
    New-Item -ItemType Directory -Path $stagingRoot, $distRoot, $workRoot | Out-Null

    $previousPythonPath = $env:PYTHONPATH
    $cpuOverlay = Join-Path $DesktopBuildDir 'cpu-overlay'
    try {
        if ($Variant -eq 'cpu') {
            if (Test-Path -LiteralPath $cpuOverlay) { Remove-Item -LiteralPath $cpuOverlay -Recurse -Force }
            New-Item -ItemType Directory -Path $cpuOverlay -Force | Out-Null
            Write-Step 'Preparing isolated CPU Torch overlay (the CUDA development environment is unchanged)'
            foreach ($line in @(Invoke-Checked -FilePath $UvExe -ArgumentList @(
                'pip', 'install', '--python', $VenvPython, '--target', $cpuOverlay,
                '--index-strategy', 'unsafe-best-match',
                '--index-url', 'https://download.pytorch.org/whl/cpu',
                '--extra-index-url', 'https://pypi.org/simple',
                'torch==2.10.0+cpu', 'torchvision==0.25.0+cpu'
            ) -WorkingDirectory $ServerDir)) { Write-Host $line }
            $env:PYTHONPATH = "$cpuOverlay;$(Join-Path $RepoRoot 'app')"
        }
        else {
            $env:PYTHONPATH = Join-Path $RepoRoot 'app'
        }

        & $VenvPython -c 'import PyInstaller' *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Step 'Installing pinned PyInstaller build tooling into the existing server environment'
            foreach ($line in @(Invoke-Checked -FilePath $UvExe -ArgumentList @('pip', 'install', '--python', $VenvPython, 'pyinstaller==6.16.0') -WorkingDirectory $ServerDir)) { Write-Host $line }
        }
        Write-Step "Freezing $Variant backend with PyInstaller"
        foreach ($line in @(Invoke-Checked -FilePath $VenvPython -ArgumentList @(
            $DesktopPythonScript, '--spec', $DesktopSpec, '--distpath', $distRoot, '--workpath', $workRoot
        ) -WorkingDirectory $RepoRoot)) { Write-Host $line }
    }
    finally {
        if ($null -eq $previousPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $previousPythonPath }
    }

    $frozenBackend = Join-Path $distRoot 'XREPORT-backend'
    $frozenExecutable = Join-Path $frozenBackend 'XREPORT-backend.exe'
    if (-not (Test-Path -LiteralPath $frozenExecutable)) { throw "PyInstaller did not produce $frozenExecutable" }
    # Copy the onedir container as a directory so PyInstaller's `_internal`
    # layout and its Python DLL dependency graph remain intact.
    Copy-Item -LiteralPath $frozenBackend -Destination (Join-Path $stagingRoot 'backend') -Recurse -Force
    $stagedBackend = Join-Path $stagingRoot 'backend'
    foreach ($directory in @(Get-ChildItem -LiteralPath $stagedBackend -Directory -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -in @('__pycache__', '.pytest_cache', '.ruff_cache', 'tests', 'test') -or
        $_.Name -match '^(pytest|playwright|ruff|pyright|jupyter|notebook|pip|setuptools|uv)([-.].*)?\.dist-info$'
    } | Sort-Object FullName -Descending)) {
        Remove-Item -LiteralPath $directory.FullName -Recurse -Force
    }
    foreach ($file in @(Get-ChildItem -LiteralPath $stagedBackend -File -Recurse -Force -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in @('.pyc', '.pyo') })) {
        Remove-Item -LiteralPath $file.FullName -Force
    }
    Copy-Item -LiteralPath $FrontendDist -Destination (Join-Path $stagingRoot 'client') -Recurse -Force
    New-Item -ItemType Directory -Path (Join-Path $stagingRoot 'settings') -Force | Out-Null
    Copy-Item -LiteralPath $EnvExample -Destination (Join-Path $stagingRoot 'settings\.env.example') -Force
    Copy-Item -LiteralPath (Join-Path $RepoRoot 'settings\configurations.json') -Destination (Join-Path $stagingRoot 'settings\configurations.json') -Force
    Copy-Item -LiteralPath (Join-Path $RepoRoot 'settings\inference_models.json') -Destination (Join-Path $stagingRoot 'settings\inference_models.json') -Force
    return $stagingRoot
}

function Add-DesktopRuntimeOverlay {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string]$Archive
    )
    # A CUDA archive is too large for include_bytes!/rustc and for an
    # in-memory PowerShell read.  Append it as a PE overlay with a fixed
    # footer; the Rust shell seeks directly to that bounded ZIP region.
    $temporary = "$Executable.runtime-overlay.tmp"
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    $source = $null
    $archiveStream = $null
    $output = $null
    try {
        $source = [IO.File]::OpenRead($Executable)
        $archiveStream = [IO.File]::OpenRead($Archive)
        $output = [IO.File]::Open($temporary, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $source.CopyTo($output, 4MB)
        $archiveOffset = [UInt64]$output.Position
        $archiveStream.CopyTo($output, 4MB)
        $archiveLength = [UInt64]($output.Position - [Int64]$archiveOffset)
        $magic = [Text.Encoding]::ASCII.GetBytes('XRPZIP01')
        $output.Write($magic, 0, $magic.Length)
        $offsetBytes = [BitConverter]::GetBytes($archiveOffset)
        $lengthBytes = [BitConverter]::GetBytes($archiveLength)
        $output.Write($offsetBytes, 0, $offsetBytes.Length)
        $output.Write($lengthBytes, 0, $lengthBytes.Length)
    }
    finally {
        if ($null -ne $output) { $output.Dispose() }
        if ($null -ne $archiveStream) { $archiveStream.Dispose() }
        if ($null -ne $source) { $source.Dispose() }
    }
    Move-Item -LiteralPath $temporary -Destination $Executable -Force
}

function Invoke-DesktopVariantBuild {
    param(
        [Parameter(Mandatory = $true)][string]$Variant,
        [Parameter(Mandatory = $true)][string]$SourceCommit,
        [Parameter(Mandatory = $true)][bool]$DirtyTree,
        [Parameter(Mandatory = $true)][string]$FrontendDist,
        [ValidateSet('Portable', 'Msi', 'All')][string]$Target = $DesktopTarget,
        [string]$ReleaseVersion = $Version
    )
    $stagingRoot = Invoke-DesktopBackendFreeze -Variant $Variant -SourceCommit $SourceCommit -FrontendDist $FrontendDist
    $archivePath = Join-Path $DesktopTauriDir 'generated\runtime.zip'
    $auditPath = Join-Path $RepoRoot "assets\QA\desktop\runtime-$Variant-$ReleaseVersion.json"
    New-Item -ItemType Directory -Path (Split-Path -Parent $archivePath), (Split-Path -Parent $auditPath) -Force | Out-Null
    $bundleArgs = @(
        $DesktopBundleScript, '--staging', $stagingRoot, '--output', $archivePath,
        '--version', $ReleaseVersion, '--variant', $Variant, '--source-commit', $SourceCommit, '--audit', $auditPath
    )
    if ($DirtyTree) { $bundleArgs += '--dirty' }
    Invoke-Checked -FilePath $VenvPython -ArgumentList $bundleArgs -WorkingDirectory $RepoRoot

    $configPath = Get-DesktopConfigPath -Variant $Variant -ReleaseVersion $ReleaseVersion
    $env:XREPORT_DESKTOP_VARIANT = $Variant
    $releaseTarget = Join-Path $DesktopTauriDir 'target\release'
    $msiDir = Join-Path $releaseTarget 'bundle\msi'
    if (Test-Path -LiteralPath $msiDir) { Remove-Item -LiteralPath $msiDir -Recurse -Force }
    $buildArgs = @('exec', '--', 'tauri', 'build', '--config', $configPath)
    if ($Target -eq 'Msi' -or $Target -eq 'All') { $buildArgs += @('--bundles', 'msi') } else { $buildArgs += '--no-bundle' }
    Write-Step "Building Tauri $Variant release"
    Invoke-Checked -FilePath $NpmCmd -ArgumentList $buildArgs -WorkingDirectory $DesktopDir

    New-Item -ItemType Directory -Path $DesktopReleaseDir -Force | Out-Null
    $portablePath = Join-Path $DesktopReleaseDir "XREPORT-v$ReleaseVersion-windows-x64-$Variant-portable.exe"
    $rawExe = Join-Path $releaseTarget 'xreport-desktop.exe'
    if (-not (Test-Path -LiteralPath $rawExe)) { throw "Expected Tauri executable not found: $rawExe" }
    $fileVersion = (Get-Item -LiteralPath $rawExe).VersionInfo.FileVersion
    if (-not $fileVersion -or $fileVersion -notlike "$ReleaseVersion*") { throw "Tauri executable version mismatch or missing file metadata: $fileVersion" }
    if ($Target -eq 'Portable' -or $Target -eq 'All') {
        Add-DesktopRuntimeOverlay -Executable $rawExe -Archive $archivePath
        Copy-Item -LiteralPath $rawExe -Destination $portablePath -Force
    }

    $msiPath = Join-Path $DesktopReleaseDir "XREPORT-v$ReleaseVersion-windows-x64-$Variant.msi"
    if ($Target -eq 'Msi' -or $Target -eq 'All') {
        $variantToken = if ($Variant -eq 'cpu') { '(?i)cpu' } else { '(?i)cuda' }
        $candidates = @(Get-ChildItem -LiteralPath $msiDir -File -Filter '*.msi' | Where-Object { $_.Name -match [regex]::Escape($ReleaseVersion) -and $_.Name -match '(?i)xreport' -and $_.Name -match $variantToken })
        if ($candidates.Count -ne 1) { throw "Expected exactly one versioned $Variant MSI; found $($candidates.Count): $($candidates.Name -join ', ')" }
        Copy-Item -LiteralPath $candidates[0].FullName -Destination $msiPath -Force
    }

    $artifactPaths = @()
    if (Test-Path -LiteralPath $portablePath) { $artifactPaths += $portablePath }
    if (Test-Path -LiteralPath $msiPath) { $artifactPaths += $msiPath }
    if ($artifactPaths.Count -eq 0) { throw "No $Variant desktop artifacts were produced" }
    $checksumPath = Join-Path $DesktopReleaseDir "XREPORT-v$ReleaseVersion-windows-x64-$Variant.sha256"
    $checksumLines = foreach ($artifact in $artifactPaths) {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifact).Hash.ToLowerInvariant()
        "$hash  $([IO.Path]::GetFileName($artifact))"
    }
    $checksumLines | Set-Content -LiteralPath $checksumPath -Encoding ascii
    $metadataPath = Join-Path $DesktopReleaseDir "XREPORT-v$ReleaseVersion-windows-x64-$Variant-build.json"
    [pscustomobject]@{
        application = 'XREPORT'
        version = $ReleaseVersion
        variant = $Variant
        source_commit = $SourceCommit
        dirty_tree = $DirtyTree
        webview2 = if ($OfflineWebView2) { 'offlineInstaller' } else { 'embedBootstrapper' }
        artifacts = @($artifactPaths | ForEach-Object { [IO.Path]::GetFileName($_) })
        checksums = [IO.Path]::GetFileName($checksumPath)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadataPath -Encoding utf8
    Write-Ok "$Variant desktop artifacts written under $DesktopReleaseDir"
}

function Invoke-BuildDesktopRelease {
    param(
        [string[]]$Variants,
        [ValidateSet('Portable', 'Msi', 'All')][string]$Target = $DesktopTarget,
        [string]$ReleaseVersion = $Version
    )
    $selectedVariants = if ($Variants -and $Variants.Count -gt 0) { @($Variants) } else { @(Get-DesktopVariants) }
    $invalidVariants = @($selectedVariants | Where-Object { $_ -notin @('cpu', 'cuda') })
    if ($invalidVariants.Count -gt 0) { throw "Unsupported desktop runtime variant: $($invalidVariants -join ', ')" }

    Assert-DesktopVersion -ExpectedVersion $ReleaseVersion
    $sourceState = Assert-DesktopSourceState
    if ($Force -and (Test-Path -LiteralPath $DesktopReleaseDir)) { Remove-Item -LiteralPath $DesktopReleaseDir -Recurse -Force }
    Ensure-PortableRuntimes
    if (-not (Test-DependenciesReady)) {
        Install-Dependencies -Settings (Import-XReportEnvironment) -InstallationType 'Standard'
    }
    $frontendDist = Invoke-DesktopFrontendBuild
    foreach ($variant in $selectedVariants) {
        Invoke-DesktopVariantBuild -Variant $variant -SourceCommit $sourceState.Commit -DirtyTree $sourceState.Dirty -FrontendDist $frontendDist -Target $Target -ReleaseVersion $ReleaseVersion
    }
    Remove-Item Env:XREPORT_DESKTOP_VARIANT -ErrorAction SilentlyContinue
    Write-Ok 'Desktop release build completed. Unsigned artifacts require WebView2 on the target machine.'
}

function Get-DesktopArtifactDefinitions {
    param([string]$ReleaseVersion = $Version)
    $prefix = "XREPORT-v$ReleaseVersion-windows-x64"
    @(
        [pscustomobject]@{
            Key = 'CpuPortable'
            Label = 'CPU portable executable'
            Variant = 'cpu'
            Target = 'Portable'
            Path = (Join-Path $DesktopReleaseDir "$prefix-cpu-portable.exe")
        }
        [pscustomobject]@{
            Key = 'CpuMsi'
            Label = 'CPU MSI installer'
            Variant = 'cpu'
            Target = 'Msi'
            Path = (Join-Path $DesktopReleaseDir "$prefix-cpu.msi")
        }
        [pscustomobject]@{
            Key = 'CudaPortable'
            Label = 'CUDA portable executable'
            Variant = 'cuda'
            Target = 'Portable'
            Path = (Join-Path $DesktopReleaseDir "$prefix-cuda-portable.exe")
        }
        [pscustomobject]@{
            Key = 'CudaMsi'
            Label = 'CUDA MSI installer'
            Variant = 'cuda'
            Target = 'Msi'
            Path = (Join-Path $DesktopReleaseDir "$prefix-cuda.msi")
        }
    )
}

function Read-DesktopReleaseVersion {
    param([ValidateSet('Create', 'Remove')][string]$Operation)
    $candidate = ([string](Read-Host "Release version to $Operation [$Version]")).Trim()
    if ([string]::IsNullOrWhiteSpace($candidate)) { return $Version }
    if ($candidate -notmatch '^\d+\.\d+\.\d+$') {
        throw "Invalid release version: $candidate. Use semantic version format such as 1.0.0."
    }
    return $candidate
}

function Read-DesktopArtifactSelection {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('Create', 'Remove')][string]$Operation,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion
    )
    $definitions = @(Get-DesktopArtifactDefinitions -ReleaseVersion $ReleaseVersion)
    while ($true) {
        Clear-Host
        Write-Host ''
        Write-Host "  DESKTOP RELEASE / $Operation" -ForegroundColor Cyan
        Write-Host "  Version: v$ReleaseVersion" -ForegroundColor DarkGray
        Write-MenuRule
        for ($index = 0; $index -lt $definitions.Count; $index++) {
            $definition = $definitions[$index]
            if ($Operation -eq 'Remove') {
                $state = if (Test-Path -LiteralPath $definition.Path) { 'present' } else { 'not found' }
                $description = "$state; update variant manifests"
            }
            else {
                $description = 'Build or rebuild this package'
            }
            Write-MenuItem -Number ([string]($index + 1)) -Label $definition.Label -Description $description
        }
        Write-Host ''
        Write-MenuItem -Number ([string]($definitions.Count + 1)) -Label 'All desktop artifacts' -Description "${Operation} all four packages" -NumberColor Green
        Write-MenuItem -Number 'B' -Label 'Back' -Description 'Return to the main menu' -NumberColor DarkGray
        Write-Host ''

        $selection = ([string](Read-Host "  Select an artifact (1-$($definitions.Count + 1), B)")).Trim()
        if ($selection -match '^B$') { return $null }
        if ($selection -eq [string]($definitions.Count + 1)) { return $definitions }
        if ($selection -match '^\d+$') {
            $index = [int]$selection - 1
            if ($index -ge 0 -and $index -lt $definitions.Count) {
                return @($definitions[$index])
            }
        }
        Write-Warn 'Invalid artifact selection.'
        [void](Read-Host 'Press Enter to continue')
    }
}

function Invoke-CreateDesktopArtifactsMenu {
    $releaseVersion = Read-DesktopReleaseVersion -Operation 'Create'
    $selected = Read-DesktopArtifactSelection -Operation 'Create' -ReleaseVersion $releaseVersion
    if ($null -eq $selected) { return }

    $selectedDefinitions = @($selected)
    $plan = foreach ($variant in @($selectedDefinitions | Select-Object -ExpandProperty Variant -Unique)) {
        $variantTargets = @($selectedDefinitions | Where-Object { $_.Variant -eq $variant } | Select-Object -ExpandProperty Target -Unique)
        [pscustomobject]@{
            Variant = $variant
            Target = if ($variantTargets.Count -gt 1) { 'All' } else { $variantTargets[0] }
        }
    }
    foreach ($targetGroup in @($plan | Group-Object -Property Target)) {
        $variants = @($targetGroup.Group | Select-Object -ExpandProperty Variant)
        Write-Step "Creating $($targetGroup.Name) artifact(s) for $($variants -join ', ') at v$releaseVersion"
        Invoke-BuildDesktopRelease -Variants $variants -Target $targetGroup.Name -ReleaseVersion $releaseVersion
    }
}

function Update-DesktopVariantReleaseMetadata {
    param(
        [Parameter(Mandatory = $true)][string]$Variant,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion
    )
    $definitions = @(Get-DesktopArtifactDefinitions -ReleaseVersion $ReleaseVersion | Where-Object { $_.Variant -eq $Variant } | Sort-Object Target)
    $payloads = @($definitions | Where-Object { Test-Path -LiteralPath $_.Path })
    $prefix = "XREPORT-v$ReleaseVersion-windows-x64-$Variant"
    $checksumPath = Join-Path $DesktopReleaseDir "$prefix.sha256"
    $metadataPath = Join-Path $DesktopReleaseDir "$prefix-build.json"

    if ($payloads.Count -eq 0) {
        foreach ($sidecar in @($checksumPath, $metadataPath)) {
            if (Test-Path -LiteralPath $sidecar) { Remove-Item -LiteralPath $sidecar -Force }
        }
        return
    }

    $checksumLines = foreach ($payload in $payloads) {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $payload.Path).Hash.ToLowerInvariant()
        "$hash  $([IO.Path]::GetFileName($payload.Path))"
    }
    $checksumLines | Set-Content -LiteralPath $checksumPath -Encoding ascii

    if (Test-Path -LiteralPath $metadataPath) {
        try {
            $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
            $metadata.artifacts = @($payloads | ForEach-Object { [IO.Path]::GetFileName($_.Path) })
            $metadata.checksums = [IO.Path]::GetFileName($checksumPath)
            $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadataPath -Encoding utf8
        }
        catch {
            Write-Warn "Could not update release metadata ${metadataPath}: $($_.Exception.Message)"
        }
    }
}

function Invoke-RemoveDesktopArtifacts {
    param(
        [Parameter(Mandatory = $true)][object[]]$Selections,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion
    )
    $removed = 0
    foreach ($selection in @($Selections)) {
        if (Test-Path -LiteralPath $selection.Path) {
            Remove-Item -LiteralPath $selection.Path -Force
            $removed++
            Write-Info "Removed $([IO.Path]::GetFileName($selection.Path))"
        }
        else {
            Write-Warn "Artifact not found: $($selection.Path)"
        }
    }

    foreach ($variant in @($Selections | Select-Object -ExpandProperty Variant -Unique)) {
        Update-DesktopVariantReleaseMetadata -Variant $variant -ReleaseVersion $ReleaseVersion
    }
    if ((Test-Path -LiteralPath $DesktopReleaseDir -PathType Container) -and -not (Get-ChildItem -LiteralPath $DesktopReleaseDir -Force)) {
        Remove-Item -LiteralPath $DesktopReleaseDir -Force
    }
    Write-Ok "Removed $removed selected release payload(s); remaining manifests were synchronized."
}

function Invoke-RemoveDesktopArtifactsMenu {
    $releaseVersion = Read-DesktopReleaseVersion -Operation 'Remove'
    $selected = Read-DesktopArtifactSelection -Operation 'Remove' -ReleaseVersion $releaseVersion
    if ($null -eq $selected) { return }
    Invoke-RemoveDesktopArtifacts -Selections @($selected) -ReleaseVersion $releaseVersion
}

function Invoke-LaunchDesktopDev {
    $settings = Import-XReportEnvironment
    Ensure-PortableRuntimes
    if (-not (Test-FrontendDependenciesReady)) { Install-FrontendDependencies }
    Invoke-FrontendBuild
    Stop-PortListener -Port ([int]$settings.FASTAPI_PORT)
    Stop-PortListener -Port ([int]$settings.UI_PORT)
    $backendAppPath = Join-Path $RepoRoot 'app'
    $escapedPython = $VenvPython.Replace("'", "''")
    $escapedApp = $backendAppPath.Replace("'", "''")
    $backendCommand = "& '$escapedPython' -m uvicorn server.app:app --app-dir '$escapedApp' --host $($settings.FASTAPI_HOST) --port $($settings.FASTAPI_PORT) --log-level info"
    $backendProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-NoExit', '-Command', $backendCommand) -WorkingDirectory $RepoRoot -WindowStyle Normal -PassThru
    $frontendCommand = "& '$($NpmCmd.Replace("'", "''"))' run preview -- --host $($settings.UI_HOST) --port $($settings.UI_PORT)"
    $frontendProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-NoExit', '-Command', $frontendCommand) -WorkingDirectory $ClientDir -WindowStyle Normal -PassThru
    try {
        Invoke-HealthCheck -Uri "http://$($settings.FASTAPI_HOST):$($settings.FASTAPI_PORT)/api/health" -TimeoutSeconds 60
        Invoke-HealthCheck -Uri "http://$($settings.UI_HOST):$($settings.UI_PORT)/" -TimeoutSeconds 60
        $env:XREPORT_DESKTOP_DEV = '1'
        Write-Step 'Launching the debug Tauri shell; backend and frontend consoles remain visible.'
        Invoke-Checked -FilePath $NpmCmd -ArgumentList @('exec', '--', 'tauri', 'dev', '--config', (Join-Path $DesktopTauriDir 'tauri.conf.json')) -WorkingDirectory $DesktopDir
    }
    finally {
        Remove-Item Env:XREPORT_DESKTOP_DEV -ErrorAction SilentlyContinue
        foreach ($process in @($frontendProcess, $backendProcess)) {
            if ($process -and -not $process.HasExited) { & taskkill.exe /PID $process.Id /T /F | Out-Null }
        }
    }
}

function Invoke-RemoveDesktopRelease {
    foreach ($target in @($DesktopReleaseDir, (Join-Path $DesktopBuildDir 'runtime-staging'), (Join-Path $DesktopBuildDir 'pyinstaller'), (Join-Path $DesktopBuildDir 'cpu-overlay'), $DesktopTargetDir, (Join-Path $DesktopTauriDir 'generated\runtime.zip'))) {
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    }
    Write-Ok 'Desktop release outputs, staging, and Tauri target files removed; user data was preserved.'
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

function Remove-PathBestEffort {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$Label = $Path
    )

    $removed = 0
    $skipped = 0
    if (-not (Test-Path -LiteralPath $Path -ErrorAction SilentlyContinue)) {
        return [pscustomobject]@{ Path = $Path; Removed = 0; Skipped = 0 }
    }

    $root = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($null -eq $root) {
        Write-Warn "Skipped inaccessible cache path: $Label"
        return [pscustomobject]@{ Path = $Path; Removed = 0; Skipped = 1 }
    }

    $children = @()
    if ($root.PSIsContainer) {
        $children = @(Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue |
            Sort-Object @{ Expression = 'PSIsContainer'; Descending = $false }, @{ Expression = { $_.FullName.Length }; Descending = $true })
        foreach ($child in $children) {
            if ($child.Name -eq '.gitkeep') { continue }
            $hasKeepMarker = $child.PSIsContainer -and @($children | Where-Object {
                $_.Name -eq '.gitkeep' -and
                $_.FullName.StartsWith("$($child.FullName)\", [StringComparison]::OrdinalIgnoreCase)
            }).Count -gt 0
            if ($hasKeepMarker) { continue }
            try {
                if ($child.PSIsContainer) {
                    Remove-Item -LiteralPath $child.FullName -Recurse -Force -ErrorAction Stop
                } else {
                    Remove-Item -LiteralPath $child.FullName -Force -ErrorAction Stop
                }
                $removed++
            } catch {
                $skipped++
                if ($skipped -le 5) {
                    Write-Warn "Skipped locked or protected cache item: $($child.FullName)"
                }
            }
        }
        if (@($children | Where-Object { $_.Name -eq '.gitkeep' }).Count -gt 0) {
            return [pscustomobject]@{ Path = $Path; Removed = $removed; Skipped = $skipped }
        }
    }

    if (Test-Path -LiteralPath $Path -ErrorAction SilentlyContinue) {
        try {
            if ($root.PSIsContainer) {
                Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            } else {
                Remove-Item -LiteralPath $Path -Force -ErrorAction Stop
            }
            $removed++
        } catch {
            $skipped++
            if ($skipped -le 5) {
                Write-Warn "Skipped locked or protected cache path: $Label"
            }
        }
    }

    [pscustomobject]@{ Path = $Path; Removed = $removed; Skipped = $skipped }
}

function Get-LegacyCacheDirectories {
    $legacyNames = @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.pyright')
    $excludedNames = @('.git', '.venv', 'node_modules', 'dist', 'build', 'release', 'target')
    $resourcesRoot = Join-Path $RepoRoot 'app\resources'
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($RepoRoot)
    $found = @()
    while ($pending.Count -gt 0) {
        $current = $pending.Pop()
        try {
            $childDirectories = @([IO.Directory]::EnumerateDirectories($current))
        } catch {
            continue
        }
        foreach ($childPath in $childDirectories) {
            $childName = Split-Path -Leaf $childPath
            if ($childName -in $excludedNames -or
                $childPath.Equals($resourcesRoot, [StringComparison]::OrdinalIgnoreCase) -or
                $childPath.StartsWith("$resourcesRoot\", [StringComparison]::OrdinalIgnoreCase)) {
                continue
            }
            if ($childName -in $legacyNames) {
                $found += Get-Item -LiteralPath $childPath -Force -ErrorAction SilentlyContinue
                continue
            }
            $pending.Push([string]$childPath)
        }
    }
    @($found | Sort-Object FullName -Descending)
}

function Remove-PythonCaches {
    $caches = @(Get-LegacyCacheDirectories | Where-Object { $_.Name -eq '__pycache__' })
    $results = @($caches | ForEach-Object { Remove-PathBestEffort -Path $_.FullName -Label $_.FullName })
    $removed = [int](($results | Measure-Object -Property Removed -Sum).Sum)
    $skipped = [int](($results | Measure-Object -Property Skipped -Sum).Sum)
    if ($skipped -gt 0) {
        Write-Warn "Removed $removed Python cache item(s); skipped $skipped protected item(s)."
    } else {
        Write-Ok "Removed $removed Python cache item(s)"
    }
}

function Clear-ApplicationCache {
    $targets = @(
        $RuntimeCacheDir,
        $ToolCacheDir,
        (Join-Path $RuntimesDir '.uv-cache'),
        (Join-Path $RepoRoot '.pytest-tmp'),
        (Join-Path $ClientDir '.angular\cache'),
        (Join-Path $ClientDir 'node_modules\.cache'),
        (Join-Path $ClientDir 'coverage')
    )
    $results = @()
    foreach ($target in $targets) {
        $results += Remove-PathBestEffort -Path $target -Label $target
    }
    foreach ($cache in Get-LegacyCacheDirectories) {
        $results += Remove-PathBestEffort -Path $cache.FullName -Label $cache.FullName
    }

    $removed = [int](($results | Measure-Object -Property Removed -Sum).Sum)
    $skipped = [int](($results | Measure-Object -Property Skipped -Sum).Sum)
    if ($skipped -gt 0) {
        Write-Warn "Application cache cleanup completed: removed $removed item(s); skipped $skipped protected item(s)."
    } else {
        Write-Ok "Application caches cleared: removed $removed item(s)."
    }
    New-Item -ItemType Directory -Path $RuntimeCacheDir, $ToolCacheDir -Force | Out-Null
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
        Remove-PathBestEffort -Path $target -Label $target | Out-Null
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
    Write-MenuItem -Number '3' -Label 'Rebuild frontend only' -Description 'Build client without launching services'

    Write-Host ''
    Write-Host '  DATA & QUALITY' -ForegroundColor DarkCyan
    Write-MenuItem -Number '4' -Label 'Initialize database' -Description 'Prepare local data store'
    Write-MenuItem -Number '5' -Label 'Run test suite' -Description 'Execute project checks'

    Write-Host ''
    Write-Host '  DESKTOP RELEASE' -ForegroundColor DarkCyan
    Write-MenuItem -Number '6' -Label 'Create release artifacts' -Description 'Build selected desktop packages'
    Write-MenuItem -Number '7' -Label 'Remove release artifacts' -Description 'Delete selected desktop packages' -NumberColor Yellow

    Write-Host ''
    Write-Host '  MAINTENANCE' -ForegroundColor DarkCyan
    Write-MenuItem -Number '8' -Label 'Remove logs' -Description 'Delete application logs'
    Write-MenuItem -Number '9' -Label 'Clear cache' -Description 'Remove temporary caches'
    Write-MenuItem -Number '10' -Label 'Uninstall application' -Description 'Remove generated files' -NumberColor Yellow

    Write-Host ''
    Write-MenuRule -Color DarkGray
    Write-MenuItem -Number '11' -Label 'Exit' -Description 'Close launcher' -NumberColor DarkGray
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
        'LaunchDesktopDev' { Invoke-LaunchDesktopDev }
        'BuildDesktopRelease' { Invoke-BuildDesktopRelease }
        'RemoveDesktopRelease' { Invoke-RemoveDesktopRelease }
        'Install' { Invoke-InstallOrUpdate }
        'RebuildFrontend' { Invoke-RebuildFrontend }
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
    $selection = (Read-Host '  Select an option (1-11)').Trim()
    if ($selection -notmatch '^(?:[1-9]|10|11)$') {
        Write-Warn 'Invalid option. Enter a number from 1 to 11.'
        [void](Read-Host 'Press Enter to continue')
        continue
    }
    if ($selection -eq '11') { break }

    try {
        switch ($selection) {
            '1' { Invoke-Launch; exit 0 }
            '2' { Invoke-InstallOrUpdate }
            '3' { Invoke-RebuildFrontend }
            '4' { Invoke-InitializeDatabase }
            '5' { Invoke-TestSuite }
            '6' { Invoke-CreateDesktopArtifactsMenu }
            '7' { Invoke-RemoveDesktopArtifactsMenu }
            '8' { Remove-Logs }
            '9' { Clear-ApplicationCache }
            '10' { Uninstall-Application }
        }
    } catch {
        Write-Fatal $_.Exception.Message
    }
    Write-Host 'Press any key to return to menu...'
    [void]$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
}
