[CmdletBinding()]
param(
    [ValidateSet('Launch', 'LaunchDesktopDev', 'BuildDesktopRelease', 'RemoveDesktopRelease', 'Install', 'RebuildFrontend', 'InitializeDatabase', 'Test', 'RemoveLogs', 'ClearCache', 'RemoveCheckpoints', 'RemoveAllData', 'Uninstall', 'KillProcesses', 'Update')]
    [string]$Action,
    [switch]$Launch,
    [ValidateSet('Cpu', 'Cuda', 'All')]
    [string]$DesktopRuntime = 'All',
    [ValidateSet('Portable', 'Msi', 'All')]
    [string]$DesktopTarget = 'All',
    [switch]$OfflineWebView2,
    [string]$Version,
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
$HuggingFaceCacheDir = Join-Path $RuntimeCacheDir 'huggingface'
$HuggingFaceHubCacheDir = Join-Path $HuggingFaceCacheDir 'hub'
$HuggingFaceModulesCacheDir = Join-Path $HuggingFaceCacheDir 'modules'
$HuggingFaceDatasetsCacheDir = Join-Path $HuggingFaceCacheDir 'datasets'
$TorchCacheDir = Join-Path $RuntimeCacheDir 'torch'
$KerasCacheDir = Join-Path $RuntimeCacheDir 'keras'
$MatplotlibCacheDir = Join-Path $RuntimeCacheDir 'matplotlib'
$PytestCacheDir = Join-Path $RuntimeCacheDir 'pytest'
$PytestTempDir = Join-Path $RuntimeCacheDir 'pytest-tmp'
$RuffCacheDir = Join-Path $RuntimeCacheDir 'ruff'
$MypyCacheDir = Join-Path $RuntimeCacheDir 'mypy'
$PythonCacheDir = Join-Path $RuntimeCacheDir 'python'
$CoverageCacheDir = Join-Path $RuntimeCacheDir 'coverage'
$AngularCacheDir = Join-Path $RuntimeCacheDir 'angular'
$CanonicalCacheDirectories = @(
    $RuntimeCacheDir,
    $UvCacheDir,
    $NpmCacheDir,
    $PipCacheDir,
    $PlaywrightBrowsersCacheDir,
    $HuggingFaceCacheDir,
    $HuggingFaceHubCacheDir,
    $HuggingFaceModulesCacheDir,
    $HuggingFaceDatasetsCacheDir,
    $TorchCacheDir,
    $KerasCacheDir,
    $MatplotlibCacheDir,
    $PytestCacheDir,
    $PytestTempDir,
    $RuffCacheDir,
    $MypyCacheDir,
    $PythonCacheDir,
    $CoverageCacheDir,
    $AngularCacheDir
)
$NodeDir = Join-Path $RuntimesDir 'nodejs'
$NodeExe = Join-Path $NodeDir 'node.exe'
$NpmCmd = Join-Path $NodeDir 'npm.cmd'
$ServerDir = Join-Path $RepoRoot 'app\server'
$ClientDir = Join-Path $RepoRoot 'app\client'
$FrontendServerScript = Join-Path $ClientDir 'scripts\serve-built.cjs'
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
$DesktopRuntimeVerifier = Join-Path $DesktopBuildDir 'verify_runtime_bundle.py'
$DesktopCpuRequirements = Join-Path $DesktopBuildDir 'cpu-runtime-requirements.txt'
$DesktopArchitecture = 'windows-x64'

if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = (Select-String -LiteralPath (Join-Path $ServerDir 'pyproject.toml') -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches.Groups[1].Value
    if ([string]::IsNullOrWhiteSpace($Version)) { throw 'Could not read the canonical version from app/server/pyproject.toml.' }
}

$PythonVersion = '3.14.7'
$PythonArchive = "python-$PythonVersion-embed-amd64.zip"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonArchive"
$PythonSha256 = 'd297e5ff019966817ad8502465176139f2d3d840fa4ed84b13bed399a6ab1f15'
$UvVersion = '0.11.9'
$UvUrlAmd64 = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
$UvUrlArm64 = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-aarch64-pc-windows-msvc.zip"
$UvSha256Amd64 = 'facbf9637c373761a96fa63c537d6c46581d357a65af01eacfd8c6319e6fb14e'
$UvSha256Arm64 = '93de7822f6214c704ec15db1b4d33eabd3709a0303ec068723d9f5f5aa99e9e7'
$NodeVersion = '22.22.3'
$NodeArchive = "node-v$NodeVersion-win-x64.zip"
$NodeUrl = "https://nodejs.org/dist/v$NodeVersion/$NodeArchive"
$NodeSha256 = '6c8d54f635feff4df76c2ca80f45332eb2ff57d25226edce36592e51a177ee33'
$NpmVersion = '10.9.8'
$FrontendBuildStateSchemaVersion = 1
$RustVersion = '1.95.0'
$script:NextProgressId = 1
$script:ActiveProgressActivities = [Collections.Generic.Dictionary[int, string]]::new()
$script:LauncherInteractive = -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected

function Write-Step([string]$Message) { Clear-LauncherProgress; Write-Host "[STEP] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Clear-LauncherProgress; Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Info([string]$Message) { Clear-LauncherProgress; Write-Host "[INFO] $Message" -ForegroundColor Gray }
function Write-Warn([string]$Message) { Clear-LauncherProgress; Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Fatal([string]$Message) { Clear-LauncherProgress; Write-Host "[FATAL] $Message" -ForegroundColor Red }

function Confirm-DestructiveAction([string]$Description) {
    if (-not $script:LauncherInteractive) {
        throw "The destructive action '$Description' requires an interactive console; no files were changed."
    }
    Clear-LauncherProgress
    $confirmation = ([string](Read-Host "Continue to $($Description)? [y/N]")).Trim()
    if ($confirmation -notmatch '^(?i:y|yes)$') {
        Write-Info 'Operation cancelled. No changes were made.'
        return $false
    }
    return $true
}

function Start-LauncherProgress {
    param([Parameter(Mandatory = $true)][string]$Activity, [Parameter(Mandatory = $true)][string]$Status)
    $id = $script:NextProgressId++
    $script:ActiveProgressActivities[$id] = $Activity
    if ($script:LauncherInteractive) { Write-Progress -Id $id -Activity $Activity -Status $Status }
    return $id
}

function Update-LauncherProgress {
    param(
        [Parameter(Mandatory = $true)][int]$Id,
        [Parameter(Mandatory = $true)][string]$Activity,
        [Parameter(Mandatory = $true)][string]$Status,
        [Nullable[int]]$PercentComplete
    )
    if (-not $script:ActiveProgressActivities.ContainsKey($Id)) { return }
    $activity = $script:ActiveProgressActivities[$Id]
    $progress = @{ Id = $Id; Activity = $activity; Status = $Status }
    if ($null -ne $PercentComplete) { $progress.PercentComplete = $PercentComplete }
    if ($script:LauncherInteractive) { Write-Progress @progress }
}

function Complete-LauncherProgress {
    param([int]$Id)
    if ($script:ActiveProgressActivities.ContainsKey($Id)) {
        $activity = $script:ActiveProgressActivities[$Id]
        try {
            if ($script:LauncherInteractive) {
                try { Write-Progress -Id $Id -Activity $activity -Completed } catch { }
            }
        }
        finally {
            [void]$script:ActiveProgressActivities.Remove($Id)
        }
    }
}

function Clear-LauncherProgress {
    foreach ($id in @($script:ActiveProgressActivities.Keys)) {
        Complete-LauncherProgress -Id $id
    }
}

function Invoke-TrackedLauncherAction {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Operation
    )
    Write-Step "Starting $Name"
    try {
        & $Operation
        Write-Ok "$Name completed"
    } catch [System.OperationCanceledException] {
        Write-Info "$Name cancelled: $($_.Exception.Message)"
    } catch {
        Write-Fatal "$Name failed: $($_.Exception.Message)"
        throw
    }
    finally {
        Clear-LauncherProgress
    }
}

function Assert-MainUpdateCheckout {
    Push-Location $RepoRoot
    try {
        $branchOutput = @(& git branch --show-current 2>$null)
        $branchExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        $branch = (@($branchOutput | ForEach-Object { [string]$_ }) -join [Environment]::NewLine).Trim()
        if ($branchExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($branch)) {
            throw 'Update requires a non-detached Git checkout.'
        }
        if ($branch -ne 'main') {
            throw "Update requires the main branch to be checked out; current branch is '$branch'. No files were changed."
        }
        $statusOutput = @(& git status --porcelain 2>$null)
        $statusExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        if ($statusExitCode -ne 0) { throw 'Unable to inspect the Git working tree before updating.' }
        $changes = @($statusOutput | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
        if ($changes.Count -gt 0) {
            throw 'Update requires a clean Git working tree. Commit or safely preserve local changes before retrying.'
        }
    } finally {
        Pop-Location
    }
}

function Invoke-Update {
    Assert-MainUpdateCheckout
    Write-Step 'Updating application from origin/main (fast-forward only).'
    Push-Location $RepoRoot
    try {
        $pullOutput = @(& git pull --ff-only origin main 2>&1)
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    } finally {
        Pop-Location
    }
    foreach ($line in $pullOutput) { Write-Host "  $line" }
    if ($exitCode -ne 0) { throw "Git update failed with exit code $exitCode." }
    Write-Ok 'Application update from origin/main completed.'
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = $RepoRoot
    )

    $display = "$FilePath " + ($ArgumentList -join ' ')
    Write-Step "Running $display"
    Push-Location $WorkingDirectory
    try {
        & $FilePath @ArgumentList
    }
    finally {
        Pop-Location
    }
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    if ($exitCode -ne 0) { throw "$FilePath failed with exit code $exitCode." }
    Write-Ok "Completed $display"
}

function Initialize-Environment {
    New-Item -ItemType Directory -Path $CanonicalCacheDirectories -Force | Out-Null
    $env:XREPORT_CACHE_ROOT = $RuntimeCacheDir
    $env:XDG_CACHE_HOME = $RuntimeCacheDir
    $env:UV_CACHE_DIR = $UvCacheDir
    $env:PIP_CACHE_DIR = $PipCacheDir
    $env:NPM_CONFIG_CACHE = $NpmCacheDir
    $env:npm_config_cache = $NpmCacheDir
    $env:PLAYWRIGHT_BROWSERS_PATH = $PlaywrightBrowsersCacheDir
    $env:PYTEST_CACHE_DIR = $PytestCacheDir
    $env:PYTEST_BASETEMP = $PytestTempDir
    $env:RUFF_CACHE_DIR = $RuffCacheDir
    $env:MYPY_CACHE_DIR = $MypyCacheDir
    $env:PYTHONPYCACHEPREFIX = $PythonCacheDir
    $env:COVERAGE_FILE = Join-Path $CoverageCacheDir '.coverage'
    $env:HF_HOME = $HuggingFaceCacheDir
    $env:HF_HUB_CACHE = $HuggingFaceHubCacheDir
    $env:HF_MODULES_CACHE = $HuggingFaceModulesCacheDir
    $env:HF_DATASETS_CACHE = $HuggingFaceDatasetsCacheDir
    $env:TORCH_HOME = $TorchCacheDir
    $env:KERAS_HOME = $KerasCacheDir
    $env:MPLCONFIGDIR = $MatplotlibCacheDir
    $env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'
    Remove-Item Env:HF_CACHE_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:TRANSFORMERS_CACHE -ErrorAction SilentlyContinue
    $env:UV_PROJECT_ENVIRONMENT = $VenvDir
    $env:UV_LINK_MODE = 'copy'
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $PythonExe) {
        # Keep venv Python extension modules aligned with the bundled
        # embeddable interpreter instead of a hosted system Python.
        $env:PYTHONHOME = $PythonDir
    }
    $env:PATH = "$NodeDir;$($env:PATH)"
}

function Invoke-DownloadAndExtract {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath,
        [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedSha256
    )
    $ProgressPreference = 'SilentlyContinue'
    $activity = "XREPORT: download and extract $([IO.Path]::GetFileName($ArchivePath))"
    $progressId = Start-LauncherProgress -Activity $activity -Status "Downloading $Uri"
    try {
        New-Item -ItemType Directory -Path (Split-Path -Parent $ArchivePath) -Force | Out-Null
        New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
        Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $ArchivePath
        Update-LauncherProgress -Id $progressId -Activity $activity -Status 'Hashing downloaded archive'
        $actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $ExpectedSha256.ToLowerInvariant()) {
            throw "Downloaded archive hash mismatch for $Uri. Expected $ExpectedSha256; got $actualSha256."
        }
        Update-LauncherProgress -Id $progressId -Activity $activity -Status 'Extracting archive'
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $DestinationPath -Force
    } finally {
        [void](Remove-LauncherPath -Path $ArchivePath -PreserveNames @() -Activity 'XREPORT: remove downloaded archive')
        Complete-LauncherProgress -Id $progressId
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
    $activity = "XREPORT: wait for health $Uri"
    $progressId = Start-LauncherProgress -Activity $activity -Status "Waiting up to $TimeoutSeconds seconds"
    try {
        do {
            $elapsed = [int](([DateTime]::Now - $deadline.AddSeconds(-$TimeoutSeconds)).TotalSeconds)
            Update-LauncherProgress -Id $progressId -Activity $activity -Status "Waiting for healthy response; ${elapsed}s elapsed"
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
    } finally {
        Complete-LauncherProgress -Id $progressId
    }
}

function Ensure-PortableRuntimes {
    param([switch]$IncludeRust)

    Write-Step 'Preparing portable runtimes'
    New-Item -ItemType Directory -Path $RuntimesDir, $PythonDir, $UvDir, $NodeDir -Force | Out-Null

    $pythonReady = $false
    if (Test-Path -LiteralPath $PythonExe) {
        $pythonOutput = (& $PythonExe --version 2>&1 | Out-String).Trim()
        $pythonReady = $pythonOutput -match "^Python $([regex]::Escape($PythonVersion))$"
    }
    if (-not $pythonReady) {
        if (Test-Path -LiteralPath $PythonDir) {
            [void](Remove-LauncherPath -Path $PythonDir -KeepRoot -PreserveNames @() -Activity 'XREPORT: replace portable Python runtime' -Strict)
        }
        Write-Info "Downloading Python $PythonVersion"
        Invoke-DownloadAndExtract -Uri $PythonUrl -ArchivePath (Join-Path $PythonDir $PythonArchive) -DestinationPath $PythonDir -ExpectedSha256 $PythonSha256
    }
    Invoke-PatchPythonPath -Path $PythonPth
    $foundVersion = Get-PythonVersion -PythonExe $PythonExe
    if ($foundVersion.Trim() -ne $PythonVersion) {
        throw "Portable Python version mismatch. Expected $PythonVersion; found $($foundVersion.Trim())."
    }
    Write-Ok "Python ready: $foundVersion"

    if (Test-Path -LiteralPath $VenvPython) {
        $venvVersion = (& $VenvPython --version 2>&1 | Out-String).Trim()
        if ($venvVersion -ne "Python $PythonVersion") {
            Write-Info "Recreating project virtual environment for Python $PythonVersion"
            [void](Remove-LauncherPath -Path $VenvDir -Activity 'XREPORT: replace project Python environment' -Strict)
        }
    }

    $uvArchitecture = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'amd64' }
    $uvExpectedVersion = "uv $UvVersion"
    $uvReady = $false
    if (Test-Path -LiteralPath $UvExe) {
        $uvOutput = (& $UvExe --version 2>&1 | Out-String).Trim()
        $uvReady = $uvOutput -like "$uvExpectedVersion*"
    }
    if (-not $uvReady) {
        if (Test-Path -LiteralPath $UvDir) {
            [void](Remove-LauncherPath -Path $UvDir -KeepRoot -PreserveNames @() -Activity 'XREPORT: replace portable uv runtime' -Strict)
        }
        $uvUrl = if ($uvArchitecture -eq 'arm64') { $UvUrlArm64 } else { $UvUrlAmd64 }
        $uvHash = if ($uvArchitecture -eq 'arm64') { $UvSha256Arm64 } else { $UvSha256Amd64 }
        Write-Info "Downloading uv $UvVersion"
        Invoke-DownloadAndExtract -Uri $uvUrl -ArchivePath (Join-Path $UvDir "uv-$UvVersion-$uvArchitecture.zip") -DestinationPath $UvDir -ExpectedSha256 $uvHash
        $foundUv = Find-UvExecutable -SearchRoot $UvDir
        if ($foundUv -ne $UvExe) {
            Copy-Item -LiteralPath $foundUv -Destination $UvExe -Force
        }
    }
    $foundUvVersion = (& $UvExe --version 2>&1 | Out-String).Trim()
    if ($foundUvVersion -notlike "$uvExpectedVersion*") {
        throw "uv version mismatch. Expected $UvVersion; found $foundUvVersion."
    }
    Write-Ok "uv ready: $foundUvVersion"

    Ensure-PortableNodeRuntime
    Initialize-Environment
    if ($IncludeRust) {
        Ensure-RustToolchain
    }
}

function Ensure-PortableNodeRuntime {
    New-Item -ItemType Directory -Path $NodeDir -Force | Out-Null
    $portableNodeNeedsUpgrade = $false
    $existingNodeVersion = $null
    $existingNpmVersion = $null
    if (Test-Path -LiteralPath $NodeExe) {
        $existingNodeVersion = (& $NodeExe --version 2>&1).TrimStart('v')
        if (Test-Path -LiteralPath $NpmCmd) {
            $existingNpmVersion = (& $NpmCmd --version 2>&1).Trim()
        }
        try {
            $portableNodeNeedsUpgrade = ([version]$existingNodeVersion -ne [version]$NodeVersion) -or ($existingNpmVersion -ne $NpmVersion)
        } catch {
            $portableNodeNeedsUpgrade = $true
        }
    }
    if (-not (Test-Path -LiteralPath $NodeExe) -or $portableNodeNeedsUpgrade) {
        if ($portableNodeNeedsUpgrade) {
            Write-Info "Upgrading portable Node.js from $existingNodeVersion to $NodeVersion"
            [void](Remove-LauncherPath -Path $NodeDir -KeepRoot -PreserveNames @() -Activity 'XREPORT: replace portable Node.js runtime' -Strict)
        }
        Write-Info "Downloading Node.js $NodeVersion"
        Invoke-DownloadAndExtract -Uri $NodeUrl -ArchivePath (Join-Path $NodeDir $NodeArchive) -DestinationPath $NodeDir -ExpectedSha256 $NodeSha256
    }
    $nestedNodeDir = Join-Path $NodeDir "node-v$NodeVersion-win-x64"
    if (Test-Path -LiteralPath (Join-Path $nestedNodeDir 'node.exe')) {
        Get-ChildItem -LiteralPath $nestedNodeDir -Force | Move-Item -Destination $NodeDir -Force
        [void](Remove-LauncherPath -Path $nestedNodeDir -Activity 'XREPORT: flatten Node.js runtime' -Strict)
    }
    if (-not (Test-Path -LiteralPath $NodeExe) -or -not (Test-Path -LiteralPath $NpmCmd)) {
        throw "Portable Node.js or npm was not found in $NodeDir."
    }
    $nodeVersionOutput = & $NodeExe --version
    $npmVersionOutput = (& $NpmCmd --version 2>&1).Trim()
    if ($nodeVersionOutput.TrimStart('v').Trim() -ne $NodeVersion -or $npmVersionOutput -ne $NpmVersion) {
        throw "Portable Node.js toolchain mismatch. Expected Node $NodeVersion/npm $NpmVersion; found $($nodeVersionOutput.Trim())/$npmVersionOutput."
    }
    Write-Ok "Node.js ready: $nodeVersionOutput"
}

function Ensure-RustToolchain {
    $rustup = Get-Command rustup.exe -ErrorAction SilentlyContinue
    if ($null -eq $rustup) {
        throw "Rust $RustVersion requires rustup.exe and the Windows MSVC Build Tools/SDK. Install rustup, then rerun the desktop build."
    }
    Invoke-Checked -FilePath $rustup.Source -ArgumentList @(
        'toolchain', 'install', $RustVersion, '--profile', 'minimal', '--component', 'rustfmt', '--component', 'clippy', '--no-self-update'
    )
    $env:RUSTUP_TOOLCHAIN = $RustVersion
    $rustOutput = (& $rustup.Source 'run' $RustVersion 'rustc' '--version' 2>&1 | Out-String).Trim()
    if ($rustOutput -notmatch "rustc $([regex]::Escape($RustVersion))(?:\s|$)") {
        throw "Rust toolchain mismatch. Expected $RustVersion; found $rustOutput."
    }
    Write-Ok "Rust ready: $rustOutput"
}

function Import-XReportEnvironment {
    $processResourceOverride = [string]$env:XREPORT_RESOURCES_DIR
    $values = @{
        FASTAPI_HOST = '127.0.0.1'
        FASTAPI_PORT = '5003'
        UI_HOST = '127.0.0.1'
        UI_PORT = '8003'
        UI_API_BASE_URL = '/api'
        RELOAD = 'false'
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
        # Keep disposable process-level resource roots ahead of project settings.
        if ($key -eq 'XREPORT_RESOURCES_DIR' -and -not [string]::IsNullOrWhiteSpace($processResourceOverride)) {
            $value = $processResourceOverride
        }
        if ($key) {
            $values[$key] = $value
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }
    return $values
}

function Install-BackendDependencies {
    param(
        [ValidateSet('Standard', 'Development', 'Desktop')]
        [string]$InstallationType = 'Standard'
    )

    Write-Step 'Synchronizing Python dependencies'
    $syncArgs = @('sync', '--frozen', '--python', $PythonExe)
    if ($InstallationType -eq 'Development') {
        $syncArgs += '--all-extras'
    }
    elseif ($InstallationType -eq 'Desktop') {
        $syncArgs += @('--extra', 'desktop')
    }
    try {
        Invoke-Checked -FilePath $UvExe -ArgumentList $syncArgs -WorkingDirectory $ServerDir
    } catch {
        Write-Warn 'Recreating the project virtual environment after a failed sync'
        [void](Remove-LauncherPath -Path $VenvDir -Activity 'XREPORT: recreate Python environment' -Strict)
        Invoke-Checked -FilePath $UvExe -ArgumentList $syncArgs -WorkingDirectory $ServerDir
    }

    if ($InstallationType -eq 'Desktop') {
        if (-not (Test-Path -LiteralPath $VenvPython)) {
            throw 'The desktop Python environment was not created by dependency synchronization.'
        }
        Write-Info 'Re-synchronizing and refreshing the locked desktop extra in the project environment'
        Invoke-Checked -FilePath $UvExe -ArgumentList @(
            'sync', '--frozen', '--python', $VenvPython, '--extra', 'desktop',
            '--reinstall-package', 'pyinstaller',
            '--reinstall-package', 'pyinstaller-hooks-contrib'
        ) -WorkingDirectory $ServerDir
        $pyInstallerProbeScript = @'
import importlib.util
import sys
import traceback

print(f"python={sys.executable}")
print(f"sys.path={sys.path}")
for module_name in ("PyInstaller", "win32ctypes", "win32ctypes.pywin32"):
    try:
        print(f"{module_name}={importlib.util.find_spec(module_name)}")
    except BaseException as exception:
        print(f"{module_name}_spec_error={exception!r}")
try:
    import PyInstaller
except BaseException:
    traceback.print_exc()
    raise
print(f"PyInstaller={PyInstaller.__version__}")
'@
        $pyInstallerProbe = (& $VenvPython -s -c $pyInstallerProbeScript 2>&1 | Out-String).Trim()
        $pyInstallerReady = $LASTEXITCODE -eq 0
        if (-not $pyInstallerReady) {
            Write-Info 'Reconciling the pinned PyInstaller toolchain directly in the project environment'
            Invoke-Checked -FilePath $UvExe -ArgumentList @(
                'pip', 'install', '--python', $VenvPython, '--reinstall', '--no-cache',
                'pyinstaller==6.22.2',
                'pyinstaller-hooks-contrib==2026.6',
                'altgraph==0.17.5',
                'packaging==26.2',
                'pefile==2023.2.7',
                'pywin32-ctypes==0.2.3',
                'setuptools==82.0.1'
            ) -WorkingDirectory $ServerDir
            $pyInstallerProbe = (& $VenvPython -s -c $pyInstallerProbeScript 2>&1 | Out-String).Trim()
            $pyInstallerReady = $LASTEXITCODE -eq 0
        }
        if (-not $pyInstallerReady) {
            throw "The locked desktop Python environment is missing an importable PyInstaller after dependency synchronization: $pyInstallerProbe"
        }
    }

}

function Install-Dependencies {
    param(
        [switch]$BuildFrontend,
        [switch]$Locked,
        [ValidateSet('Standard', 'Development', 'Desktop')]
        [string]$InstallationType = 'Standard'
    )

    Install-BackendDependencies -InstallationType $InstallationType
    Install-FrontendDependencies -Locked:$Locked
    if ($InstallationType -eq 'Desktop') {
        Install-DesktopDependencies -Locked:$Locked
    }

    if ($BuildFrontend) {
        Invoke-FrontendBuild
    }
}

function Install-FrontendDependencies {
    param([switch]$Locked)
    Write-Step 'Installing frontend dependencies'
    if (-not (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json'))) {
        throw "Locked frontend installation requires $ClientDir\package-lock.json."
    }
    Invoke-Checked -FilePath $NpmCmd -ArgumentList @('ci') -WorkingDirectory $ClientDir
}

function Install-DesktopDependencies {
    param([switch]$Locked)
    Write-Step 'Installing desktop dependencies'
    if (-not (Test-Path -LiteralPath (Join-Path $DesktopDir 'package-lock.json'))) {
        throw "Locked desktop installation requires $DesktopDir\package-lock.json."
    }
    Invoke-Checked -FilePath $NpmCmd -ArgumentList @('ci') -WorkingDirectory $DesktopDir
}

function Invoke-FrontendBuild {
    $beforeFingerprint = Get-FrontendBuildFingerprint
    $beforeDependencyFingerprint = Get-FrontendDependencyFingerprint
    Write-Step 'Building frontend'
    Invoke-Checked -FilePath $NpmCmd -ArgumentList @('run', 'build') -WorkingDirectory $ClientDir
    $frontendOutput = Join-Path $ClientDir 'dist\client-angular\browser\index.html'
    if (-not (Test-Path -LiteralPath $frontendOutput)) {
        throw "Angular production output was not created: $frontendOutput"
    }
    $afterFingerprint = Get-FrontendBuildFingerprint
    $afterDependencyFingerprint = Get-FrontendDependencyFingerprint
    if ($beforeFingerprint -ne $afterFingerprint -or $beforeDependencyFingerprint -ne $afterDependencyFingerprint) {
        throw 'Frontend inputs changed while the build was running. The build-state manifest was not written; rerun the frontend build.'
    }
    Write-FrontendBuildState -BuildFingerprint $afterFingerprint -DependencyFingerprint $afterDependencyFingerprint
}

function Get-FrontendProductionInputRelativePaths {
    $relativePaths = @(
        'angular.json',
        'package.json',
        'package-lock.json',
        'tsconfig.json',
        'tsconfig.app.json'
    )
    $clientRoot = [IO.Path]::GetFullPath($ClientDir)
    $separators = [char[]]@([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $clientRootPrefix = $clientRoot.TrimEnd($separators) + [IO.Path]::DirectorySeparatorChar
    foreach ($root in @('public', 'src')) {
        $rootPath = Join-Path $ClientDir $root
        if (-not (Test-Path -LiteralPath $rootPath -PathType Container)) { continue }
        foreach ($file in @(Get-ChildItem -LiteralPath $rootPath -Recurse -File)) {
            $fullPath = [IO.Path]::GetFullPath($file.FullName)
            if (-not $fullPath.StartsWith($clientRootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Frontend production input escaped the client directory: $($file.FullName)"
            }
            $relative = $fullPath.Substring($clientRootPrefix.Length).Replace('\', '/')
            if ($relative -match '(?i)^src/.+\.spec\.ts$' -or $relative -ieq 'src/proxy.conf.cjs') { continue }
            $relativePaths += $relative
        }
    }
    return @($relativePaths)
}

function Get-FrontendDependencyRelativePaths {
    return @('package.json', 'package-lock.json')
}

function Get-FrontendFingerprint {
    param(
        [Parameter(Mandatory = $true)][string[]]$RelativePaths,
        [string[]]$Context = @()
    )

    $utf8 = [Text.UTF8Encoding]::new($false)
    $stream = [IO.MemoryStream]::new()
    $hash = [Security.Cryptography.SHA256]::Create()
    try {
        $sortedContext = [string[]]@($Context)
        [Array]::Sort($sortedContext, [StringComparer]::Ordinal)
        foreach ($value in $sortedContext) {
            $contextBytes = $utf8.GetBytes("CONTEXT:$value`n")
            $stream.Write($contextBytes, 0, $contextBytes.Length)
        }
        $normalizedPaths = [string[]]@(
            $RelativePaths | ForEach-Object {
                ([string]$_).Replace('\', '/').ToLowerInvariant()
            }
        )
        [Array]::Sort($normalizedPaths, [StringComparer]::Ordinal)
        $previousPath = $null
        foreach ($normalized in $normalizedPaths) {
            if ($null -ne $previousPath -and [StringComparer]::Ordinal.Equals($normalized, $previousPath)) {
                continue
            }
            $previousPath = $normalized
            $pathBytes = $utf8.GetBytes("PATH:$normalized`n")
            $stream.Write($pathBytes, 0, $pathBytes.Length)
            $absolute = Join-Path $ClientDir ($normalized.Replace('/', '\'))
            if (Test-Path -LiteralPath $absolute -PathType Leaf) {
                $contentBytes = [IO.File]::ReadAllBytes($absolute)
                $stream.Write($contentBytes, 0, $contentBytes.Length)
            }
            else {
                $missingBytes = $utf8.GetBytes("MISSING:$normalized`n")
                $stream.Write($missingBytes, 0, $missingBytes.Length)
            }
            $endBytes = $utf8.GetBytes("END:$normalized`n")
            $stream.Write($endBytes, 0, $endBytes.Length)
        }
        return ([BitConverter]::ToString($hash.ComputeHash($stream.ToArray())) -replace '-', '').ToLowerInvariant()
    }
    finally {
        $hash.Dispose()
        $stream.Dispose()
    }
}

function Get-FrontendBuildFingerprint {
    return Get-FrontendFingerprint -RelativePaths (Get-FrontendProductionInputRelativePaths) -Context @(
        "schema=$FrontendBuildStateSchemaVersion",
        "node=$NodeVersion"
    )
}

function Get-FrontendDependencyFingerprint {
    return Get-FrontendFingerprint -RelativePaths (Get-FrontendDependencyRelativePaths) -Context @(
        "schema=$FrontendBuildStateSchemaVersion",
        "node=$NodeVersion"
    )
}

function Write-FrontendBuildState {
    param(
        [Parameter(Mandatory = $true)][string]$BuildFingerprint,
        [Parameter(Mandatory = $true)][string]$DependencyFingerprint
    )

    $stateDirectory = Join-Path $ClientDir 'dist\client-angular'
    $statePath = Join-Path $stateDirectory '.xreport-build-state.json'
    New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
    $temporaryPath = "$statePath.$PID.tmp"
    $state = [ordered]@{
        schema_version = $FrontendBuildStateSchemaVersion
        production_input_fingerprint = $BuildFingerprint
        dependency_manifest_fingerprint = $DependencyFingerprint
        node_version = $NodeVersion
        build_completed_utc = [DateTime]::UtcNow.ToString('o')
    }
    $encoding = [Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllText($temporaryPath, ($state | ConvertTo-Json -Depth 4), $encoding)
    Move-Item -LiteralPath $temporaryPath -Destination $statePath -Force
    Write-Ok "Frontend build state refreshed: $([IO.Path]::GetFileName($statePath))"
}

function Get-FrontendBuildStatus {
    $frontendOutput = Join-Path $ClientDir 'dist\client-angular\browser\index.html'
    $statePath = Join-Path $ClientDir 'dist\client-angular\.xreport-build-state.json'
    $buildFingerprint = $null
    $dependencyFingerprint = $null
    $state = $null
    $isCurrent = $false
    $reason = 'OutputMissing'

    if (Test-Path -LiteralPath $frontendOutput -PathType Leaf) {
        $buildFingerprint = Get-FrontendBuildFingerprint
        $dependencyFingerprint = Get-FrontendDependencyFingerprint
        if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
            $reason = 'ManifestMissing'
        }
        else {
            try {
                $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
                if ([int]$state.schema_version -ne $FrontendBuildStateSchemaVersion) {
                    $reason = 'SchemaChanged'
                }
                elseif ([string]$state.node_version -ne $NodeVersion) {
                    $reason = 'NodeVersionChanged'
                }
                elseif ([string]$state.dependency_manifest_fingerprint -ne $dependencyFingerprint) {
                    $reason = 'DependencyManifestChanged'
                }
                elseif ([string]$state.production_input_fingerprint -ne $buildFingerprint) {
                    $reason = 'ProductionInputChanged'
                }
                else {
                    $isCurrent = $true
                    $reason = 'Current'
                }
            }
            catch {
                $reason = 'ManifestInvalid'
            }
        }
    }

    return [pscustomobject]@{
        IsCurrent = $isCurrent
        Reason = $reason
        OutputPath = $frontendOutput
        StatePath = $statePath
        BuildFingerprint = $buildFingerprint
        DependencyFingerprint = $dependencyFingerprint
        State = $state
    }
}

function Ensure-FrontendBuild {
    param([psobject]$Status)

    $status = if ($null -eq $Status) { Get-FrontendBuildStatus } else { $Status }
    if ($status.IsCurrent) {
        Write-Ok 'Frontend production bundle is current; skipped Angular build.'
        return $status
    }

    Write-Info "Frontend production bundle requires work: $($status.Reason)"
    if ($status.Reason -eq 'DependencyManifestChanged' -or -not (Test-FrontendDependenciesReady)) {
        Install-FrontendDependencies -Locked
    }
    Invoke-FrontendBuild
    $finalStatus = Get-FrontendBuildStatus
    if (-not $finalStatus.IsCurrent) {
        throw "Frontend build completed without a current build-state manifest: $($finalStatus.Reason)"
    }
    return $finalStatus
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

function Test-PortableNodeRuntimeReady {
    if (-not (Test-Path -LiteralPath $NodeExe) -or -not (Test-Path -LiteralPath $NpmCmd)) {
        return $false
    }
    $nodeVersionOutput = (& $NodeExe --version 2>&1 | Out-String).Trim()
    $npmVersionOutput = (& $NpmCmd --version 2>&1 | Out-String).Trim()
    return $nodeVersionOutput.TrimStart('v').Trim() -eq $NodeVersion -and $npmVersionOutput -eq $NpmVersion
}

function Test-DesktopDependenciesReady {
    $desktopPackage = Join-Path $DesktopDir 'package.json'
    $desktopLock = Join-Path $DesktopDir 'package-lock.json'
    $desktopModules = Join-Path $DesktopDir 'node_modules'
    $desktopInstallState = Join-Path $desktopModules '.package-lock.json'
    $desktopRunner = Join-Path $desktopModules '.bin\tauri.cmd'

    return (Test-Path -LiteralPath $desktopPackage) -and
        (Test-Path -LiteralPath $desktopLock) -and
        (Test-Path -LiteralPath $desktopInstallState) -and
        (Test-Path -LiteralPath $desktopRunner)
}

function Test-BackendDependenciesReady {
    $backendEntrypoint = Join-Path $ServerDir 'app.py'

    if (-not (Test-Path -LiteralPath $PythonExe) -or
        -not (Test-Path -LiteralPath $UvExe) -or
        -not (Test-Path -LiteralPath $VenvPython) -or
        -not (Test-Path -LiteralPath $backendEntrypoint)) {
        return $false
    }

    $pythonVersionOutput = (& $PythonExe --version 2>&1 | Out-String).Trim()
    if ($pythonVersionOutput -ne "Python $PythonVersion") { return $false }
    & $UvExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    $venvVersionOutput = (& $VenvPython --version 2>&1 | Out-String).Trim()
    if ($venvVersionOutput -ne "Python $PythonVersion") { return $false }
    & $VenvPython -c 'import fastapi, uvicorn' *> $null
    if ($LASTEXITCODE -ne 0) { return $false }

    return $true
}

function Stop-PortListener {
    param(
        [Parameter(Mandatory = $true)][int]$Port,
        [int[]]$ExcludeProcessIds = @()
    )

    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $ExcludeProcessIds -notcontains [int]$_.OwningProcess } |
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

function Get-PortConflicts {
    param(
        [Parameter(Mandatory = $true)][int]$FastApiPort,
        [Parameter(Mandatory = $true)][int]$UiPort,
        [object[]]$ProcessTable = @()
    )

    $configuredPorts = @($FastApiPort, $UiPort) | Sort-Object -Unique
    $connections = @()
    $connectionLookupError = $null
    try {
        $connections = @(Get-NetTCPConnection -State Listen -ErrorAction Stop |
            Where-Object { $configuredPorts -contains [int]$_.LocalPort -and [int]$_.OwningProcess -gt 0 })
    }
    catch {
        $connectionLookupError = $_.Exception.Message
    }

    if ($connections.Count -eq 0) {
        try {
            $netstatOutput = @(& netstat.exe -ano -p tcp 2>&1)
            if ($LASTEXITCODE -ne 0) {
                throw "netstat.exe exited with code $LASTEXITCODE."
            }
            $connections = @(
                foreach ($line in $netstatOutput) {
                    if ([string]$line -match '^\s*TCP\s+\S+:(?<localPort>\d+)\s+\S+\s+LISTENING\s+(?<processId>\d+)\s*$') {
                        $localPort = [int]$Matches.localPort
                        $processId = [int]$Matches.processId
                        if ($configuredPorts -contains $localPort -and $processId -gt 0) {
                            [pscustomobject]@{ LocalPort = $localPort; OwningProcess = $processId }
                        }
                    }
                }
            )
        }
        catch {
            $netstatError = $_.Exception.Message
            throw "Unable to inspect configured TCP listeners. Get-NetTCPConnection: $connectionLookupError; netstat.exe: $netstatError"
        }
    }
    if ($connections.Count -eq 0) { return @() }
    if ($ProcessTable.Count -eq 0) {
        try { $ProcessTable = @(Get-XReportProcessTable) } catch { $ProcessTable = @() }
    }

    $conflictsByPid = @{}
    foreach ($connection in $connections) {
        $processId = [int]$connection.OwningProcess
        if (-not $conflictsByPid.ContainsKey($processId)) {
            $conflictsByPid[$processId] = [ordered]@{ ProcessId = $processId; Ports = @() }
        }
        $ports = @(@($conflictsByPid[$processId].Ports) + @([int]$connection.LocalPort) | Sort-Object -Unique)
        $conflictsByPid[$processId].Ports = $ports
    }

    $conflicts = foreach ($processId in @($conflictsByPid.Keys | Sort-Object)) {
        $process = @($ProcessTable | Where-Object { [int]$_.ProcessId -eq $processId } | Select-Object -First 1)
        [pscustomobject]@{
            ProcessId = $processId
            ProcessName = if ($process.Count -gt 0 -and $process[0].Name) { [string]$process[0].Name } else { '<unavailable>' }
            ExecutablePath = if ($process.Count -gt 0) { [string]$process[0].ExecutablePath } else { '' }
            CommandLine = if ($process.Count -gt 0) { [string]$process[0].CommandLine } else { '' }
            Ports = @($conflictsByPid[$processId].Ports)
        }
    }
    return @($conflicts)
}

function Get-ProtectedLauncherProcessIds {
    param([Parameter(Mandatory = $true)][object[]]$ProcessTable)

    $protectedProcessIds = @([int]$PID)
    $currentProcessId = [int]$PID
    while ($true) {
        $currentProcess = @($ProcessTable | Where-Object { [int]$_.ProcessId -eq $currentProcessId } | Select-Object -First 1)
        if ($currentProcess.Count -eq 0) { break }
        $parentProcessId = [int]$currentProcess[0].ParentProcessId
        if ($parentProcessId -le 0 -or $protectedProcessIds -contains $parentProcessId) { break }
        $protectedProcessIds += $parentProcessId
        $currentProcessId = $parentProcessId
    }
    return @($protectedProcessIds | Sort-Object -Unique)
}

function Format-PortConflict {
    param([Parameter(Mandatory = $true)][psobject]$Conflict)

    $ports = (@($Conflict.Ports | Sort-Object) -join ', ')
    $description = "PID $($Conflict.ProcessId) ($($Conflict.ProcessName)) holds port(s) $ports"
    if (-not [string]::IsNullOrWhiteSpace([string]$Conflict.ExecutablePath)) {
        $description += "; executable $($Conflict.ExecutablePath)"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$Conflict.CommandLine)) {
        $description += "; command $($Conflict.CommandLine)"
    }
    return $description
}

function Wait-ForConfiguredPortsAvailable {
    param(
        [Parameter(Mandatory = $true)][int]$FastApiPort,
        [Parameter(Mandatory = $true)][int]$UiPort,
        [int]$Attempts = 40
    )

    for ($attempt = 0; $attempt -lt $Attempts; $attempt++) {
        $conflicts = @(Get-PortConflicts -FastApiPort $FastApiPort -UiPort $UiPort)
        if ($conflicts.Count -eq 0) { return @() }
        Start-Sleep -Milliseconds 250
    }
    return @(Get-PortConflicts -FastApiPort $FastApiPort -UiPort $UiPort)
}

function Resolve-LaunchPortConflicts {
    param([Parameter(Mandatory = $true)][hashtable]$Settings)

    $fastApiPort = 0
    $uiPort = 0
    if (-not [int]::TryParse([string]$Settings.FASTAPI_PORT, [ref]$fastApiPort) -or $fastApiPort -lt 1 -or $fastApiPort -gt 65535) {
        throw "FASTAPI_PORT must be an integer between 1 and 65535; found '$($Settings.FASTAPI_PORT)'."
    }
    if (-not [int]::TryParse([string]$Settings.UI_PORT, [ref]$uiPort) -or $uiPort -lt 1 -or $uiPort -gt 65535) {
        throw "UI_PORT must be an integer between 1 and 65535; found '$($Settings.UI_PORT)'."
    }
    if ($fastApiPort -eq $uiPort) {
        throw "FASTAPI_PORT and UI_PORT must be different; both are configured as $fastApiPort."
    }

    $processTable = @()
    $conflicts = @(Get-PortConflicts -FastApiPort $fastApiPort -UiPort $uiPort)
    if ($conflicts.Count -eq 0) {
        Write-Ok "Launch ports $fastApiPort and $uiPort are available."
        return
    }
    try { $processTable = @(Get-XReportProcessTable) }
    catch { Write-Warn "Process metadata is unavailable; conflicts will be displayed by PID and port only: $($_.Exception.Message)" }
    if ($processTable.Count -gt 0) {
        $conflicts = @(Get-PortConflicts -FastApiPort $fastApiPort -UiPort $uiPort -ProcessTable $processTable)
    }

    Write-Warn 'Configured launch ports are occupied:'
    foreach ($conflict in $conflicts) { Write-Host "  $(Format-PortConflict -Conflict $conflict)" -ForegroundColor Yellow }

    if (-not $script:LauncherInteractive) {
        throw 'Launch ports are occupied and this invocation is non-interactive; no process was terminated. Free the listed ports or rerun interactively.'
    }
    if ($processTable.Count -eq 0) {
        throw 'Process metadata is unavailable for the occupied launch ports; no process was terminated. Free the ports and rerun.'
    }
    $protectedProcessIds = @(Get-ProtectedLauncherProcessIds -ProcessTable $processTable)
    $protectedConflicts = @($conflicts | Where-Object { $protectedProcessIds -contains [int]$_.ProcessId })
    if ($protectedConflicts.Count -gt 0) {
        throw "A launcher or ancestor process owns a configured port. Stop it explicitly before launching: $(($protectedConflicts | ForEach-Object { "PID $($_.ProcessId)" }) -join ', ')."
    }

    Clear-LauncherProgress
    $confirmation = ([string](Read-Host 'Terminate all listed processes once and continue? [y/N]')).Trim()
    if ($confirmation -notmatch '^(?i:y|yes)$') {
        throw [System.OperationCanceledException]::new('Launch cancelled by the user. No process was terminated and no services were started.')
    }

    $approvedProcessIds = @($conflicts | ForEach-Object { [int]$_.ProcessId } | Sort-Object -Unique)
    foreach ($processId in $approvedProcessIds) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            Write-Info "Approved PID $processId exited before termination; treating it as resolved."
            continue
        }
        $taskkillOutput = @(& taskkill.exe /PID $processId /T /F 2>&1)
        $taskkillExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        if ($taskkillExitCode -ne 0) {
            if ($null -eq (Get-Process -Id $processId -ErrorAction SilentlyContinue)) { continue }
            $detail = (@($taskkillOutput | ForEach-Object { [string]$_ }) -join ' ').Trim()
            $approvedConflict = @($conflicts | Where-Object { [int]$_.ProcessId -eq $processId } | Select-Object -First 1)
            $approvedName = if ($approvedConflict.Count -gt 0) { [string]$approvedConflict[0].ProcessName } else { '<unavailable>' }
            throw "Unable to terminate approved PID $processId ($approvedName); taskkill exit code $taskkillExitCode. $detail"
        }
    }

    $remainingConflicts = @(Wait-ForConfiguredPortsAvailable -FastApiPort $fastApiPort -UiPort $uiPort)
    if ($remainingConflicts.Count -gt 0) {
        $unapproved = @($remainingConflicts | Where-Object { $approvedProcessIds -notcontains [int]$_.ProcessId })
        if ($unapproved.Count -gt 0) {
            throw "A new or unapproved process now owns a configured launch port; it was not terminated: $(($unapproved | ForEach-Object { Format-PortConflict -Conflict $_ }) -join ' | ')"
        }
        throw "Approved process(es) still occupy configured launch ports after termination: $(($remainingConflicts | ForEach-Object { Format-PortConflict -Conflict $_ }) -join ' | ')"
    }
    Write-Ok 'Configured launch ports are available after approved conflict resolution.'
}

function Get-XReportProcessTable {
    try {
        return @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop |
            Select-Object ProcessId, ParentProcessId, Name, ExecutablePath, CommandLine)
    }
    catch {
        throw "Unable to inspect Windows processes for XREPORT cleanup: $($_.Exception.Message)"
    }
}

function Get-XReportApplicationProcessIds {
    param([Parameter(Mandatory = $true)][object[]]$ProcessTable)

    $repoPattern = [regex]::Escape(([IO.Path]::GetFullPath($RepoRoot)).TrimEnd('\'))
    $processIds = foreach ($process in $ProcessTable) {
        $commandLine = [string]$process.CommandLine
        $executablePath = [string]$process.ExecutablePath
        $processName = [IO.Path]::GetFileNameWithoutExtension([string]$process.Name)
        $repoScoped = ($commandLine -match $repoPattern) -or ($executablePath -match $repoPattern)
        $isBackend = $repoScoped -and ($commandLine -match '(?i)(?:server\.app:app|\buvicorn\b)')
        $isFrontend = $repoScoped -and ($commandLine -match '(?i)(?:\bnpm\b|\bnode(?:\.exe)?\b|\bvite\b).*(?:\bpreview\b|serve-built\.cjs)')
        $isDesktopDevelopment = $repoScoped -and ($commandLine -match '(?i)\b(?:tauri|cargo)\b')
        $isPackagedApplication = $processName -in @('xreport-backend', 'xreport-desktop')

        if ($isBackend -or $isFrontend -or $isDesktopDevelopment -or $isPackagedApplication) {
            [int]$process.ProcessId
        }
    }
    return @($processIds | Sort-Object -Unique)
}

function Get-XReportProcessTreeIds {
    param(
        [Parameter(Mandatory = $true)][object[]]$ProcessTable,
        [Parameter(Mandatory = $true)][int[]]$RootProcessIds
    )

    $processIds = @($RootProcessIds | ForEach-Object { [int]$_ } | Sort-Object -Unique)
    do {
        $childProcessIds = @(
            foreach ($process in $ProcessTable) {
                $processId = [int]$process.ProcessId
                $parentProcessId = [int]$process.ParentProcessId
                if ($processId -gt 0 -and $processIds -contains $parentProcessId -and $processIds -notcontains $processId) {
                    $processId
                }
            }
        )
        $nextProcessIds = @($processIds + $childProcessIds | Sort-Object -Unique)
        $changed = $nextProcessIds.Count -gt $processIds.Count
        $processIds = $nextProcessIds
    } while ($changed)

    return $processIds
}

function Stop-XReportProcesses {
    $settings = Import-XReportEnvironment
    $processTable = Get-XReportProcessTable
    $protectedProcessIds = @(Get-ProtectedLauncherProcessIds -ProcessTable $processTable)

    $configuredPorts = @($settings.FASTAPI_PORT, $settings.UI_PORT) |
        ForEach-Object { [int]$_ } |
        Sort-Object -Unique
    $rootProcessIds = @(Get-XReportApplicationProcessIds -ProcessTable $processTable)
    foreach ($port in $configuredPorts) {
        $portProcessId = Get-PortProcessId -Port $port
        if ($null -ne $portProcessId) { $rootProcessIds += [int]$portProcessId }
    }
    $rootProcessIds = @(
        $rootProcessIds |
            Where-Object { $protectedProcessIds -notcontains [int]$_ } |
            ForEach-Object { [int]$_ } |
            Sort-Object -Unique
    )

    if ($rootProcessIds.Count -eq 0) {
        foreach ($port in $configuredPorts) {
            Stop-PortListener -Port $port -ExcludeProcessIds $protectedProcessIds
        }
        Write-Info 'No XREPORT application processes or configured listeners were found.'
        return
    }

    $treeProcessIds = @(
        Get-XReportProcessTreeIds -ProcessTable $processTable -RootProcessIds $rootProcessIds |
            Where-Object { $protectedProcessIds -notcontains [int]$_ } |
            ForEach-Object { [int]$_ } |
            Sort-Object -Unique
    )
    Write-Info "Stopping $($treeProcessIds.Count) XREPORT process(es)."

    foreach ($rootProcessId in $rootProcessIds) {
        $processInfo = @($processTable | Where-Object { [int]$_.ProcessId -eq $rootProcessId } | Select-Object -First 1)
        $processName = if ($processInfo.Count -gt 0) { [string]$processInfo[0].Name } else { 'process' }
        Write-Info "Stopping $processName (PID $rootProcessId)"
        $null = & taskkill.exe /PID $rootProcessId /T /F 2>$null
    }

    $remainingProcessIds = @(
        $treeProcessIds | Where-Object {
            $null -ne (Get-Process -Id ([int]$_) -ErrorAction SilentlyContinue)
        }
    )
    foreach ($processId in ($remainingProcessIds | Sort-Object -Descending)) {
        try { Stop-Process -Id ([int]$processId) -Force -ErrorAction Stop } catch { }
    }

    foreach ($port in $configuredPorts) {
        Stop-PortListener -Port $port -ExcludeProcessIds $protectedProcessIds
    }

    $remainingProcessIds = @(
        $treeProcessIds | Where-Object {
            $null -ne (Get-Process -Id ([int]$_) -ErrorAction SilentlyContinue)
        }
    )
    if ($remainingProcessIds.Count -gt 0) {
        throw "Unable to stop XREPORT process(es): $($remainingProcessIds -join ', ')."
    }
    Write-Ok 'All XREPORT application processes and configured listeners were stopped.'
}

function Invoke-Launch {
    $settings = Import-XReportEnvironment
    $launchStopwatch = [Diagnostics.Stopwatch]::StartNew()
    $portPreflightStarted = $launchStopwatch.ElapsedMilliseconds
    Resolve-LaunchPortConflicts -Settings $settings
    Write-Info "Launch timing phase=port_preflight elapsed_ms=$($launchStopwatch.ElapsedMilliseconds - $portPreflightStarted)"

    $dependencyStarted = $launchStopwatch.ElapsedMilliseconds
    Initialize-Environment
    if (-not (Test-BackendDependenciesReady)) {
        Write-Step 'Backend environment is missing or unusable; installing backend dependencies.'
        Ensure-PortableRuntimes
        Install-BackendDependencies -InstallationType 'Standard'
    }
    else {
        Write-Ok 'Backend environment is ready; skipped backend dependency installation.'
    }
    if (-not (Test-PortableNodeRuntimeReady)) {
        Ensure-PortableNodeRuntime
    }
    Write-Info "Launch timing phase=dependency_readiness elapsed_ms=$($launchStopwatch.ElapsedMilliseconds - $dependencyStarted)"

    $buildCheckStarted = $launchStopwatch.ElapsedMilliseconds
    $buildStatus = Get-FrontendBuildStatus
    Write-Info "Frontend build status: $($buildStatus.Reason)"
    Write-Info "Launch timing phase=build_freshness_check elapsed_ms=$($launchStopwatch.ElapsedMilliseconds - $buildCheckStarted)"
    $rebuildStarted = $launchStopwatch.ElapsedMilliseconds
    Ensure-FrontendBuild -Status $buildStatus | Out-Null
    Write-Info "Launch timing phase=stale_rebuild elapsed_ms=$($launchStopwatch.ElapsedMilliseconds - $rebuildStarted)"

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Virtual-environment Python was not found at $VenvPython."
    }
    $backendAppPath = Join-Path $RepoRoot 'app'
    Write-Step 'Starting backend'
    $escapedPython = $VenvPython.Replace("'", "''")
    $escapedApp = $backendAppPath.Replace("'", "''")
    $backendCommand = "& '$escapedPython' -m uvicorn server.app:app --app-dir '$escapedApp' --host $($settings.FASTAPI_HOST) --port $($settings.FASTAPI_PORT) --log-level info"
    if ($settings.RELOAD -eq 'true') { $backendCommand += ' --reload' }
    $backendStartStarted = $launchStopwatch.ElapsedMilliseconds
    $backendProcess = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @('-NoProfile', '-NoExit', '-Command', $backendCommand) `
        -WorkingDirectory $RepoRoot -WindowStyle Normal -PassThru
    Write-Info "Launch timing phase=backend_process_start elapsed_ms=$($launchStopwatch.ElapsedMilliseconds - $backendStartStarted)"

    $healthUrl = "http://$($settings.FASTAPI_HOST):$($settings.FASTAPI_PORT)/api/health"
    $uiUrl = "http://$($settings.UI_HOST):$($settings.UI_PORT)"
    $frontendProcess = $null
    try {
        if (-not (Test-Path -LiteralPath $FrontendServerScript)) {
            throw "Built frontend server was not found: $FrontendServerScript"
        }
        Write-Step 'Starting built frontend server while the backend initializes'
        $frontendStartStarted = $launchStopwatch.ElapsedMilliseconds
        $quotedFrontendServerScript = '"' + $FrontendServerScript + '"'
        $frontendProcess = Start-Process -FilePath $NodeExe -ArgumentList @(
            $quotedFrontendServerScript,
            '--host', $settings.UI_HOST,
            '--port', $settings.UI_PORT,
            '--api-base-url', $settings.UI_API_BASE_URL,
            '--backend-host', $settings.FASTAPI_HOST,
            '--backend-port', $settings.FASTAPI_PORT
        ) `
            -WorkingDirectory $ClientDir -WindowStyle Hidden -PassThru
        Write-Info "Launch timing phase=frontend_server_start elapsed_ms=$($launchStopwatch.ElapsedMilliseconds - $frontendStartStarted)"
        Write-Step "Waiting for frontend at $uiUrl"
        $uiReachabilityStarted = $launchStopwatch.ElapsedMilliseconds
        Invoke-HealthCheck -Uri "$uiUrl/" -TimeoutSeconds 60
        Write-Info "Launch timing phase=ui_reachable elapsed_ms=$($launchStopwatch.ElapsedMilliseconds - $uiReachabilityStarted)"

        try {
            Start-Process -FilePath $uiUrl -ErrorAction Stop | Out-Null
        }
        catch {
            Write-Warn "Automatic browser launch failed. Open the interface manually at $uiUrl. $($_.Exception.Message)"
        }

        Write-Ok 'XREPORT interface started. Backend initialization is continuing in the application.'
        Write-Host "Backend: $healthUrl (initializing; launcher PID $($backendProcess.Id))"
        Write-Host "Frontend: $uiUrl (PID $($frontendProcess.Id))"
    }
    catch {
        if ($frontendProcess -and -not $frontendProcess.HasExited) {
            & taskkill.exe /PID $frontendProcess.Id /T /F | Out-Null
        }
        if ($backendProcess -and -not $backendProcess.HasExited) {
            & taskkill.exe /PID $backendProcess.Id /T /F | Out-Null
        }
        throw
    }
}

function Invoke-InstallOrUpdate {
    Ensure-PortableRuntimes
    Write-Ok 'Portable runtimes ready.'
    $installationType = Read-InstallationType
    $settings = Import-XReportEnvironment
    Stop-PortListener -Port ([int]$settings.UI_PORT)
    Install-Dependencies -BuildFrontend -InstallationType $installationType
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
    $backendVersion = $serverVersion
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
    $capabilityPath = Join-Path $DesktopTauriDir 'capabilities\default.json'
    $capability = Get-Content -LiteralPath $capabilityPath -Raw | ConvertFrom-Json
    if ($config.app.security.PSObject.Properties.Name -contains 'capabilities') {
        $config.app.security.capabilities = @($capability)
    }
    else {
        $config.app.security | Add-Member -MemberType NoteProperty -Name capabilities -Value @($capability)
    }
    $config.version = $ReleaseVersion
    if ($OfflineWebView2) {
        $config.bundle.windows.webviewInstallMode = [pscustomobject]@{ type = 'offlineInstaller' }
    }
    $config | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $configPath -Encoding utf8
    return $configPath
}

function Invoke-DesktopFrontendBuild {
    param([switch]$Strict)
    if ($Strict) {
        Install-FrontendDependencies -Locked
    }
    elseif (-not (Test-FrontendDependenciesReady)) {
        Ensure-PortableNodeRuntime
        foreach ($line in @(Install-FrontendDependencies)) { Write-Host $line }
    }
    foreach ($line in @(Invoke-FrontendBuild)) { Write-Host $line }
    $frontendOutput = Join-Path $ClientDir 'dist\client-angular\browser\index.html'
    if (-not (Test-Path -LiteralPath $frontendOutput)) {
        throw "Angular production output was not created: $frontendOutput"
    }
    $frontendDist = Split-Path -Parent $frontendOutput
    $desktopUi = Join-Path $DesktopTauriDir 'ui'
    if (Test-Path -LiteralPath $desktopUi) { [void](Remove-LauncherPath -Path $desktopUi -Activity 'XREPORT: refresh desktop UI staging' -Strict) }
    New-Item -ItemType Directory -Path $desktopUi -Force | Out-Null
    foreach ($entry in @(Get-ChildItem -LiteralPath $frontendDist -Force)) {
        Copy-Item -LiteralPath $entry.FullName -Destination $desktopUi -Recurse -Force
    }
    return $frontendDist
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
    if (Test-Path -LiteralPath $stagingRoot) { [void](Remove-LauncherPath -Path $stagingRoot -Activity 'XREPORT: refresh desktop runtime staging' -Strict) }
    if (Test-Path -LiteralPath $distRoot) { [void](Remove-LauncherPath -Path $distRoot -Activity 'XREPORT: refresh PyInstaller output' -Strict) }
    if (Test-Path -LiteralPath $workRoot) { [void](Remove-LauncherPath -Path $workRoot -Activity 'XREPORT: refresh PyInstaller work area' -Strict) }
    New-Item -ItemType Directory -Path $stagingRoot, $distRoot, $workRoot | Out-Null

    $previousPythonPath = $env:PYTHONPATH
    $cpuOverlay = Join-Path $DesktopBuildDir 'cpu-overlay'
    try {
        if ($Variant -eq 'cpu') {
            if (Test-Path -LiteralPath $cpuOverlay) { [void](Remove-LauncherPath -Path $cpuOverlay -Activity 'XREPORT: refresh CPU overlay' -Strict) }
            New-Item -ItemType Directory -Path $cpuOverlay -Force | Out-Null
            Write-Step 'Preparing isolated CPU Torch overlay (the CUDA development environment is unchanged)'
            foreach ($line in @(Invoke-Checked -FilePath $UvExe -ArgumentList @(
                'pip', 'install', '--python', $VenvPython, '--target', $cpuOverlay,
                '--require-hashes', '--no-deps', '--only-binary', ':all:',
                '--index-url', 'https://download.pytorch.org/whl/cpu',
                '--requirement', $DesktopCpuRequirements
            ) -WorkingDirectory $ServerDir)) { Write-Host $line }
            $env:PYTHONPATH = "$cpuOverlay;$(Join-Path $RepoRoot 'app')"
        }
        else {
            $env:PYTHONPATH = Join-Path $RepoRoot 'app'
        }

        & $VenvPython -s -c 'import PyInstaller' *> $null
        if ($LASTEXITCODE -ne 0) {
            throw 'The locked desktop Python environment is missing PyInstaller. Re-run the desktop dependency synchronization.'
        }
        Write-Step "Freezing $Variant backend with PyInstaller"
        foreach ($line in @(Invoke-Checked -FilePath $VenvPython -ArgumentList @(
            '-s', $DesktopPythonScript, '--spec', $DesktopSpec, '--distpath', $distRoot, '--workpath', $workRoot
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
    $pruneDirectoryNames = @('__pycache__', '.pytest_cache', '.ruff_cache', '.cache', 'cache', 'caches', 'tests', 'test')
    $pruneDirectories = @(Get-ChildItem -LiteralPath $stagedBackend -Directory -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -in $pruneDirectoryNames -or
        $_.Name -match '^(pytest|playwright|ruff|pyright|jupyter|notebook|pip|setuptools|uv)([-.].*)?\.dist-info$'
    } | Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $false }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    $pruneRoots = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($directory in $pruneDirectories) {
        $ancestor = [IO.Path]::GetDirectoryName($directory.FullName)
        $covered = $false
        while (-not [string]::IsNullOrWhiteSpace($ancestor) -and $ancestor.StartsWith("$stagedBackend\", [StringComparison]::OrdinalIgnoreCase)) {
            if ($pruneRoots.Contains($ancestor)) {
                $covered = $true
                break
            }
            $ancestor = [IO.Path]::GetDirectoryName($ancestor)
        }
        if ($covered) { continue }
        [void](Remove-LauncherPath -Path $directory.FullName -Activity "XREPORT: prune frozen backend $($directory.Name)" -Strict)
        [void]$pruneRoots.Add($directory.FullName)
    }
    foreach ($file in @(Get-ChildItem -LiteralPath $stagedBackend -File -Recurse -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -in @('.pyc', '.pyo') } |
            Sort-Object @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })) {
        [void](Remove-LauncherPath -Path $file.FullName -PreserveNames @() -Activity "XREPORT: remove frozen bytecode $($file.Name)" -Strict)
    }
    Copy-Item -LiteralPath $FrontendDist -Destination (Join-Path $stagingRoot 'client') -Recurse -Force
    New-Item -ItemType Directory -Path (Join-Path $stagingRoot 'settings') -Force | Out-Null
    Copy-Item -LiteralPath $EnvExample -Destination (Join-Path $stagingRoot 'settings\.env.example') -Force
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
    if (Test-Path -LiteralPath $temporary) {
        [void](Remove-LauncherPath -Path $temporary -PreserveNames @() -Activity 'XREPORT: remove stale runtime overlay' -Strict)
    }
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
        $artifactPrefix = Join-Path $DesktopReleaseDir "XREPORT-v$ReleaseVersion-windows-x64-$Variant"
        foreach ($staleArtifact in @(
            "${artifactPrefix}-portable.exe",
            "${artifactPrefix}.msi",
            "${artifactPrefix}.sha256",
            "${artifactPrefix}-build.json"
        )) {
            if (Test-Path -LiteralPath $staleArtifact) {
                [void](Remove-LauncherPath -Path $staleArtifact -PreserveNames @() -Activity "XREPORT: remove stale release file $([IO.Path]::GetFileName($staleArtifact))" -Strict)
            }
        }
        if (Test-Path -LiteralPath $auditPath) {
            [void](Remove-LauncherPath -Path $auditPath -PreserveNames @() -Activity 'XREPORT: remove stale runtime audit' -Strict)
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $archivePath), (Split-Path -Parent $auditPath) -Force | Out-Null
        if (Test-Path -LiteralPath $archivePath) {
            [void](Remove-LauncherPath -Path $archivePath -PreserveNames @() -Activity 'XREPORT: remove stale runtime archive' -Strict)
        }
        $bundleArgs = @(
            $DesktopBundleScript, '--staging', $stagingRoot, '--output', $archivePath,
            '--version', $ReleaseVersion, '--variant', $Variant, '--architecture', $DesktopArchitecture,
            '--source-commit', $SourceCommit, '--audit', $auditPath
        )
        if ($DirtyTree) { $bundleArgs += '--dirty' }
        Invoke-Checked -FilePath $VenvPython -ArgumentList $bundleArgs -WorkingDirectory $RepoRoot
        $runtimeManifestJson = Get-Content -LiteralPath $auditPath -Raw
        $runtimeManifest = $runtimeManifestJson | ConvertFrom-Json
        $createdUtcMatch = [regex]::Match($runtimeManifestJson, '"created_utc"\s*:\s*"([^"]+)"')
        if (-not $createdUtcMatch.Success) { throw "Runtime audit is missing a raw created_utc value: $auditPath" }
        $createdUtc = $createdUtcMatch.Groups[1].Value
        Invoke-Checked -FilePath $VenvPython -ArgumentList @(
            $DesktopRuntimeVerifier, '--archive', $archivePath, '--version', $ReleaseVersion,
            '--variant', $Variant, '--architecture', $DesktopArchitecture, '--source-commit', $SourceCommit
        ) -WorkingDirectory $RepoRoot

        $configPath = Get-DesktopConfigPath -Variant $Variant -ReleaseVersion $ReleaseVersion
        $cargoTargetRoot = Join-Path $DesktopBuildDir "cargo-target\$Variant"
        if (Test-Path -LiteralPath $cargoTargetRoot) { [void](Remove-LauncherPath -Path $cargoTargetRoot -Activity "XREPORT: refresh Cargo target $Variant" -Strict) }
        $previousVariant = $env:XREPORT_DESKTOP_VARIANT
        $previousCargoTarget = $env:CARGO_TARGET_DIR
        $env:XREPORT_DESKTOP_VARIANT = $Variant
        $env:CARGO_TARGET_DIR = $cargoTargetRoot
        $releaseTarget = Join-Path $cargoTargetRoot 'release'
        $msiDir = Join-Path $releaseTarget 'bundle\msi'
        try {
            $buildArgs = @('exec', '--', 'tauri', 'build', '--config', $configPath, '--ci', '--no-sign')
            if ($Target -eq 'Msi' -or $Target -eq 'All') { $buildArgs += @('--bundles', 'msi') } else { $buildArgs += '--no-bundle' }
            $buildArgs += @('--', '--locked')
            Write-Step "Building Tauri $Variant release"
            Invoke-Checked -FilePath $NpmCmd -ArgumentList $buildArgs -WorkingDirectory $DesktopDir
        }
        finally {
            if ($null -eq $previousVariant) { Remove-Item Env:XREPORT_DESKTOP_VARIANT -ErrorAction SilentlyContinue } else { $env:XREPORT_DESKTOP_VARIANT = $previousVariant }
            if ($null -eq $previousCargoTarget) { Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue } else { $env:CARGO_TARGET_DIR = $previousCargoTarget }
        }

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
            format = 2
            application = 'XREPORT'
            version = $ReleaseVersion
            variant = $Variant
            architecture = $DesktopArchitecture
            source_commit = $SourceCommit
            dirty_tree = $DirtyTree
            # Windows PowerShell converts ISO timestamps to DateTime values while
            # deserializing JSON. Preserve the source string so release metadata
            # stays locale-independent and matches the runtime audit exactly.
            created_utc = $createdUtc
            payload_sha256 = [string]$runtimeManifest.payload_sha256
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
        if ($Force -and (Test-Path -LiteralPath $DesktopReleaseDir)) {
            if (-not (Confirm-DestructiveAction 'replace the existing desktop release output')) { return }
            [void](Remove-LauncherPath -Path $DesktopReleaseDir -PreserveNames @() -Activity 'XREPORT: remove existing desktop release' -Strict)
        }
        foreach ($generatedPath in @(
            (Join-Path $DesktopTauriDir 'ui'),
            (Join-Path $DesktopTauriDir 'generated\runtime.zip')
        )) {
            if (Test-Path -LiteralPath $generatedPath) {
                [void](Remove-LauncherPath -Path $generatedPath -PreserveNames @() -Activity "XREPORT: remove generated release path $([IO.Path]::GetFileName($generatedPath))" -Strict)
            }
        }
        try {
            Ensure-PortableRuntimes -IncludeRust
            Install-Dependencies -Locked -InstallationType 'Desktop'
            $frontendDist = Invoke-DesktopFrontendBuild
            foreach ($variant in @($selectedVariants)) {
                Invoke-DesktopVariantBuild -Variant $variant -SourceCommit $sourceState.Commit -DirtyTree $sourceState.Dirty -FrontendDist $frontendDist -Target $Target -ReleaseVersion $ReleaseVersion
            }
        }
        finally {
            Remove-Item Env:XREPORT_DESKTOP_VARIANT -ErrorAction SilentlyContinue
        }
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
    Clear-LauncherProgress
    $candidate = ([string](Read-Host "Release version to $Operation [$Version]")).Trim()
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        Clear-LauncherProgress
        return $Version
    }
    if ($candidate -notmatch '^\d+\.\d+\.\d+$') {
        throw "Invalid release version: $candidate. Use semantic version format such as 3.1.0."
    }
    Clear-LauncherProgress
    return $candidate
}

function Read-DesktopArtifactSelection {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('Create', 'Remove')][string]$Operation,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion
    )
    $definitions = @(Get-DesktopArtifactDefinitions -ReleaseVersion $ReleaseVersion)
    while ($true) {
        Clear-LauncherProgress
        if ($script:LauncherInteractive) { try { Clear-Host } catch { } }
        Write-Host ''
        Write-Host "  DESKTOP RELEASE / $Operation" -ForegroundColor Cyan
        Write-Host "  Version: v$ReleaseVersion" -ForegroundColor DarkGray
        Write-MenuRule
        $entries = @(
            for ($index = 0; $index -lt $definitions.Count; $index++) {
                $definition = $definitions[$index]
                if ($Operation -eq 'Remove') {
                    $state = if (Test-Path -LiteralPath $definition.Path) { 'present' } else { 'not found' }
                    $description = "$state; update variant manifests"
                }
                else {
                    $description = 'Build or rebuild this package'
                }
                [pscustomobject]@{
                    Key = $definition.Key
                    Label = $definition.Label
                    Description = $description
                    Definition = $definition
                    Destructive = $Operation -eq 'Remove'
                }
            }
            [pscustomobject]@{
                Key = 'All'
                Label = 'All desktop artifacts'
                Description = "${Operation} all four packages"
                Definition = $definitions
                Destructive = $Operation -eq 'Remove'
            }
            [pscustomobject]@{
                Key = 'Back'
                Label = 'Back'
                Description = 'Return to the main menu'
                Definition = $null
                Destructive = $false
            }
        )
        for ($index = 0; $index -lt $entries.Count; $index++) {
            $entries[$index] | Add-Member -NotePropertyName Number -NotePropertyValue ($index + 1) -Force
        }
        $numberWidth = ([string]$entries.Count).Length
        $labelWidth = ($entries | ForEach-Object { $_.Label.Length } | Measure-Object -Maximum).Maximum
        foreach ($entry in $entries) {
            Write-MenuItem -Entry $entry -NumberWidth $numberWidth -LabelWidth $labelWidth
        }
        Write-Host ''

        $selection = ([string](Read-Host "  Select an option (1-$($entries.Count))")).Trim()
        $selectedNumber = 0
        if (-not [int]::TryParse($selection, [ref]$selectedNumber) -or $selectedNumber -lt 1 -or $selectedNumber -gt $entries.Count) {
            Write-Warn "Invalid option. Select a number from 1 through $($entries.Count)."
            Wait-ForMenu
            continue
        }
        $selectedEntry = $entries[$selectedNumber - 1]
        if ($selectedEntry.Key -eq 'Back') {
            Clear-LauncherProgress
            return $null
        }
        if ($selectedEntry.Key -eq 'All') {
            Clear-LauncherProgress
            return @($definitions)
        }
        Clear-LauncherProgress
        return @($selectedEntry.Definition)
    }
}

function Invoke-CreateDesktopArtifactsMenu {
    $releaseVersion = Read-DesktopReleaseVersion -Operation 'Create'
    $selected = Read-DesktopArtifactSelection -Operation 'Create' -ReleaseVersion $releaseVersion
    if ($null -eq $selected) {
        Clear-LauncherProgress
        return
    }

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
            if (Test-Path -LiteralPath $sidecar) {
                [void](Remove-LauncherPath -Path $sidecar -PreserveNames @() -Activity "XREPORT: remove stale release manifest $([IO.Path]::GetFileName($sidecar))" -Strict)
            }
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
    if (-not (Confirm-DestructiveAction 'remove the selected desktop release artifacts')) { return }
    $removed = 0
    foreach ($selection in @($Selections)) {
        if (Test-Path -LiteralPath $selection.Path) {
            $result = Remove-LauncherPath -Path $selection.Path -PreserveNames @() -Activity "XREPORT: remove release file $([IO.Path]::GetFileName($selection.Path))" -Strict
            $removed += $result.RemovedCount
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
        [void](Remove-LauncherPath -Path $DesktopReleaseDir -PreserveNames @() -Activity 'XREPORT: remove empty release directory' -Strict)
    }
    Write-Ok "Removed $removed selected release payload(s); remaining manifests were synchronized."
}

function Invoke-RemoveDesktopArtifactsMenu {
    $releaseVersion = Read-DesktopReleaseVersion -Operation 'Remove'
    $selected = Read-DesktopArtifactSelection -Operation 'Remove' -ReleaseVersion $releaseVersion
    if ($null -eq $selected) {
        Clear-LauncherProgress
        return
    }
    Invoke-RemoveDesktopArtifacts -Selections @($selected) -ReleaseVersion $releaseVersion
}

function Invoke-LaunchDesktopDev {
    $settings = Import-XReportEnvironment
    Ensure-PortableRuntimes
    if (-not (Test-FrontendDependenciesReady)) { Install-FrontendDependencies }
    if (-not (Test-DesktopDependenciesReady)) { Install-DesktopDependencies }
    Invoke-DesktopFrontendBuild | Out-Null
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
        Invoke-HealthCheck -Uri "http://$($settings.UI_HOST):$($settings.UI_PORT)/" -TimeoutSeconds 60
        $env:XREPORT_DESKTOP_DEV = '1'
        $devConfigPath = Get-DesktopConfigPath -Variant 'cpu' -ReleaseVersion $Version
        Write-Step 'Launching the debug Tauri shell; backend and frontend consoles remain visible.'
        Invoke-Checked -FilePath $NpmCmd -ArgumentList @('exec', '--', 'tauri', 'dev', '--config', $devConfigPath) -WorkingDirectory $DesktopDir
    }
    finally {
        Remove-Item Env:XREPORT_DESKTOP_DEV -ErrorAction SilentlyContinue
        foreach ($process in @($frontendProcess, $backendProcess)) {
            if ($process -and -not $process.HasExited) { & taskkill.exe /PID $process.Id /T /F | Out-Null }
        }
    }
}

function Invoke-RemoveDesktopRelease {
    if (-not (Confirm-DestructiveAction 'remove all desktop release outputs and generated release state')) { return }
    $generatedConfigs = @(Get-ChildItem -LiteralPath $DesktopBuildDir -File -Filter 'tauri-*.json' -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    $targets = @(
        $DesktopReleaseDir,
        (Join-Path $DesktopBuildDir 'runtime-staging'),
        (Join-Path $DesktopBuildDir 'pyinstaller'),
        (Join-Path $DesktopBuildDir 'cpu-overlay'),
        (Join-Path $DesktopBuildDir 'cargo-target'),
        $DesktopTargetDir,
        (Join-Path $DesktopTauriDir 'generated\runtime.zip'),
        (Join-Path $DesktopTauriDir 'ui')
    ) + $generatedConfigs
    $results = foreach ($target in $targets) {
        Remove-LauncherPath -Path $target -Activity "XREPORT: remove $target"
    }
    $skipped = [int](($results | Measure-Object -Property Skipped -Sum).Sum)
    if ($skipped -gt 0) {
        Write-Warn "Desktop release cleanup completed; skipped $skipped locked or protected item(s)."
    } else {
        Write-Ok 'Desktop release outputs, staging, and Tauri target files removed; user data was preserved.'
    }
}

function Read-InstallationType {
    Clear-LauncherProgress
    Write-Host '  [1] Development - include Ruff, Pyright, and pytest'
    Write-Host '  [2] Standard    - install runtime dependencies only'
    $selection = (Read-Host '  Select installation profile [1-2]').Trim()
    switch ($selection) {
        '1' { Clear-LauncherProgress; return 'Development' }
        '2' { Clear-LauncherProgress; return 'Standard' }
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

function Get-ConfiguredResourceRoot {
    $processResourceOverride = [string]$env:XREPORT_RESOURCES_DIR
    $settings = Import-XReportEnvironment
    $configuredRoot = if (-not [string]::IsNullOrWhiteSpace($processResourceOverride)) {
        $processResourceOverride
    } elseif ($settings.ContainsKey('XREPORT_RESOURCES_DIR')) {
        [string]$settings['XREPORT_RESOURCES_DIR']
    } else {
        'app/resources'
    }
    if ([string]::IsNullOrWhiteSpace($configuredRoot)) {
        $configuredRoot = 'app/resources'
    }
    $expandedRoot = [Environment]::ExpandEnvironmentVariables($configuredRoot.Trim())
    if (-not [IO.Path]::IsPathRooted($expandedRoot)) {
        $expandedRoot = Join-Path $RepoRoot $expandedRoot
    }
    $resourceRoot = [IO.Path]::GetFullPath($expandedRoot).TrimEnd('\')
    $filesystemRoot = ([IO.Path]::GetPathRoot($resourceRoot)).TrimEnd('\')
    $repositoryRoot = [IO.Path]::GetFullPath($RepoRoot).TrimEnd('\')
    if ([string]::IsNullOrWhiteSpace($resourceRoot) -or $resourceRoot -eq $filesystemRoot -or $resourceRoot -eq $repositoryRoot -or $repositoryRoot.StartsWith("$resourceRoot\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove data from the configured resource root '$resourceRoot'."
    }
    return $resourceRoot
}

function Get-XReportUserDataTargets {
    param([switch]$CheckpointsOnly)

    $resourceRoot = Get-ConfiguredResourceRoot
    if ($CheckpointsOnly) {
        return @((Join-Path $resourceRoot 'checkpoints'))
    }

    $databasePath = Join-Path $resourceRoot 'database.db'
    return @(
        $databasePath,
        "$databasePath-wal",
        "$databasePath-shm",
        "$databasePath-journal",
        (Join-Path $resourceRoot 'checkpoints'),
        (Join-Path $resourceRoot 'models'),
        (Join-Path $resourceRoot 'models\tokenizers'),
        (Join-Path $resourceRoot 'logs')
    ) | ForEach-Object { [IO.Path]::GetFullPath($_) } | Select-Object -Unique
}

function Remove-XReportUserDataTargets {
    param([Parameter(Mandatory = $true)][string[]]$Targets)

    $removed = 0
    $skipped = 0
    foreach ($target in @($Targets | Select-Object -Unique)) {
        $isContainer = Test-Path -LiteralPath $target -PathType Container
        $result = Remove-LauncherPath -Path $target -KeepRoot:$isContainer -Activity "XREPORT: remove user data $target"
        $removed += $result.Removed
        $skipped += $result.Skipped
    }
    return [pscustomobject]@{ Removed = $removed; Skipped = $skipped }
}

function Remove-Checkpoints {
    if (-not (Confirm-DestructiveAction 'remove all saved checkpoints')) { return }
    $result = Remove-XReportUserDataTargets -Targets @(Get-XReportUserDataTargets -CheckpointsOnly)
    if ($result.Skipped -gt 0) {
        Write-Warn "Removed $($result.Removed) checkpoint item(s); skipped $($result.Skipped) locked or protected item(s)."
    } else {
        Write-Ok "Removed $($result.Removed) checkpoint item(s)."
    }
}

function Remove-AllData {
    if (-not (Confirm-DestructiveAction 'remove all local user-generated data')) { return }
    $result = Remove-XReportUserDataTargets -Targets @(Get-XReportUserDataTargets)
    if ($result.Skipped -gt 0) {
        Write-Warn "Removed $($result.Removed) local data item(s); skipped $($result.Skipped) locked or protected item(s). External databases were not modified."
    } else {
        Write-Ok "Removed $($result.Removed) local data item(s); external databases were not modified. Application files and settings were preserved."
    }
}

function Remove-Logs {
    if (-not (Confirm-DestructiveAction 'remove application log files')) { return }
    $logDir = Join-Path (Get-ConfiguredResourceRoot) 'logs'
    $logs = @(Get-ChildItem -LiteralPath $logDir -File -Filter '*.log' -ErrorAction SilentlyContinue |
        Sort-Object @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    if ($logs.Count -gt 0) {
        $removed = 0
        $skipped = 0
        $progressId = Start-LauncherProgress -Activity 'XREPORT: remove application logs' -Status "0 of $($logs.Count) files"
        try {
            for ($index = 0; $index -lt $logs.Count; $index++) {
                $log = $logs[$index]
                Update-LauncherProgress -Id $progressId -Activity 'XREPORT: remove application logs' -Status "$($index + 1) of $($logs.Count): $($log.Name)" -PercentComplete ([int](($index + 1) * 100 / $logs.Count))
                $result = Remove-LauncherPath -Path $log.FullName -Activity "XREPORT: remove $($log.FullName)"
                $removed += $result.Removed
                $skipped += $result.Skipped
            }
        }
        finally {
            Complete-LauncherProgress -Id $progressId
        }
        if ($skipped -gt 0) {
            Write-Warn "Removed $removed log file(s); skipped $skipped locked or protected file(s)."
        } else {
            Write-Ok "Removed $removed log file(s)"
        }
    } else {
        Write-Info 'No log files found'
    }
}

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

    $fullPath = [IO.Path]::GetFullPath($Path)
    $trimmedPath = $fullPath.TrimEnd('\')
    $pathRoot = ([IO.Path]::GetPathRoot($fullPath)).TrimEnd('\')
    if ([string]::IsNullOrWhiteSpace($pathRoot) -or $trimmedPath.Equals($pathRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove filesystem root: $fullPath"
    }
    if ($RepoRoot) {
        $repoRootPath = ([IO.Path]::GetFullPath($RepoRoot)).TrimEnd('\')
        if ($trimmedPath.Equals($repoRootPath, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove the repository root: $fullPath"
        }
    }

    $plannedPaths = [Collections.Generic.List[string]]::new()
    $removedPaths = [Collections.Generic.List[string]]::new()
    $preservedPathList = [Collections.Generic.List[string]]::new()
    $skippedPaths = [Collections.Generic.List[string]]::new()
    $enumerationMessages = [Collections.Generic.List[string]]::new()
    $result = [ordered]@{
        Target = $fullPath
        Path = $fullPath
        Planned = 0
        PlannedCount = 0
        PlannedPaths = @()
        Removed = 0
        RemovedCount = 0
        RemovedPaths = @()
        Preserved = 0
        PreservedCount = 0
        PreservedEntries = @()
        PreservedPaths = @()
        Skipped = 0
        SkippedCount = 0
        SkippedPaths = @()
        EnumerationErrors = @()
        EnumerationErrorCount = 0
        WhatIf = [bool]$WhatIf
    }
    try {
        $root = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
    }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound) { return [pscustomobject]$result }
        $message = "$fullPath ($($_.Exception.Message))"
        [void]$skippedPaths.Add($message)
        [void]$enumerationMessages.Add($message)
        $result.Skipped = 1
        $result.SkippedCount = 1
        $result.SkippedPaths = @($skippedPaths)
        $result.EnumerationErrors = @($enumerationMessages)
        $result.EnumerationErrorCount = 1
        Write-Warn "Skipped inaccessible path: $fullPath ($($_.Exception.Message))"
        if ($Strict) { throw "Removal of '$fullPath' failed: $($_.Exception.Message)" }
        return [pscustomobject]$result
    }
    $enumerationErrors = @()
    $entries = if ($root.PSIsContainer) {
        @(Get-ChildItem -LiteralPath $root.FullName -Force -Recurse -ErrorAction SilentlyContinue -ErrorVariable enumerationErrors)
    } else { @($root) }
    foreach ($enumerationError in @($enumerationErrors)) {
        if ($null -eq $enumerationError) { continue }
        $message = "$fullPath ($($enumerationError.Exception.Message))"
        [void]$enumerationMessages.Add($message)
        [void]$skippedPaths.Add($message)
        Write-Warn "Skipped inaccessible path below ${fullPath}: $($enumerationError.Exception.Message)"
    }
    $preservedPathSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $protectedDirectories = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $protectedSubtreePrefixes = [Collections.Generic.List[string]]::new()
    $preserveNameSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($preserveName in @($PreserveNames)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$preserveName)) {
            [void]$preserveNameSet.Add([string]$preserveName)
        }
    }
    foreach ($entry in @($entries)) {
        if ($preserveNameSet.Contains([string]$entry.Name)) {
            [void]$preservedPathSet.Add($entry.FullName)
            [void]$preservedPathList.Add($entry.FullName)
            if ($entry.PSIsContainer) {
                [void]$protectedSubtreePrefixes.Add($entry.FullName.TrimEnd('\') + '\')
                [void]$protectedDirectories.Add($entry.FullName)
            }
            [void]$protectedDirectories.Add($root.FullName)
            $ancestor = [IO.Path]::GetDirectoryName($entry.FullName)
            while ($ancestor -and $ancestor.StartsWith($root.FullName.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
                [void]$protectedDirectories.Add($ancestor)
                $ancestor = [IO.Path]::GetDirectoryName($ancestor)
            }
        }
    }
    $candidates = @($entries |
        Where-Object {
            if ($preservedPathSet.Contains($_.FullName) -or $protectedDirectories.Contains($_.FullName)) {
                $false
            } else {
                $insideProtectedSubtree = $false
                foreach ($prefix in $protectedSubtreePrefixes) {
                    if ($_.FullName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                        $insideProtectedSubtree = $true
                        break
                    }
                }
                -not $insideProtectedSubtree
            }
        } |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    if ($root.PSIsContainer -and -not $KeepRoot -and $preservedPathList.Count -eq 0) { $candidates += $root }
    foreach ($candidate in @($candidates)) { [void]$plannedPaths.Add($candidate.FullName) }
    $result.Planned = [int]$plannedPaths.Count
    $result.PlannedCount = [int]$plannedPaths.Count
    $result.PlannedPaths = @($plannedPaths)
    $result.Preserved = [int]$preservedPathList.Count
    $result.PreservedCount = [int]$preservedPathList.Count
    $result.PreservedEntries = @($preservedPathList)
    $result.PreservedPaths = @($preservedPathList)
    $result.EnumerationErrors = @($enumerationMessages)
    $result.EnumerationErrorCount = [int]$enumerationMessages.Count
    $progressId = $null
    try {
        if ($plannedPaths.Count -gt 0) { $progressId = Start-LauncherProgress -Activity $Activity -Status "0 of $($plannedPaths.Count) items" }
        for ($index = 0; $index -lt $plannedPaths.Count; $index++) {
            $entry = $candidates[$index]
            if ($null -ne $progressId) {
                Update-LauncherProgress -Id $progressId -Activity $Activity -Status "$($index + 1) of $($plannedPaths.Count): $($entry.Name)" -PercentComplete ([int](($index + 1) * 100 / [Math]::Max(1, $plannedPaths.Count)))
            }
            if ($WhatIf) { continue }
            try {
                Remove-Item -LiteralPath $entry.FullName -Force -Recurse -Confirm:$false -ErrorAction Stop
                [void]$removedPaths.Add($entry.FullName)
            }
            catch {
                [void]$skippedPaths.Add("$($entry.FullName) ($($_.Exception.Message))")
                Write-Warn "Skipped locked or protected path: $($entry.FullName) ($($_.Exception.Message))"
            }
        }
    }
    finally {
        if ($null -ne $progressId) { Complete-LauncherProgress -Id $progressId }
    }
    $result.Removed = [int]$removedPaths.Count
    $result.RemovedCount = [int]$removedPaths.Count
    $result.RemovedPaths = @($removedPaths)
    $result.Skipped = [int]$skippedPaths.Count
    $result.SkippedCount = [int]$skippedPaths.Count
    $result.SkippedPaths = @($skippedPaths)
    if ($Strict -and ($result.SkippedCount -gt 0 -or $result.EnumerationErrorCount -gt 0)) {
        throw "Removal of '$fullPath' was incomplete. Skipped $($result.SkippedCount) item(s) and encountered $($result.EnumerationErrorCount) enumeration error(s)."
    }
    return [pscustomobject]$result
}

function Get-LegacyCacheDirectories {
    $legacyPaths = @(
        (Join-Path $RepoRoot 'app\tests\cache'),
        (Join-Path $RepoRoot 'app\server\app\tests\cache'),
        (Join-Path $RepoRoot '.pytest_cache'),
        (Join-Path $RepoRoot '.ruff_cache'),
        (Join-Path $RepoRoot '.mypy_cache'),
        (Join-Path $RepoRoot '.pyright'),
        (Join-Path $RepoRoot '.uv-cache'),
        (Join-Path $RepoRoot '.pytest-tmp'),
        (Join-Path $RuntimesDir '.uv-cache'),
        (Join-Path $ServerDir 'runtimes\cache'),
        (Join-Path $ClientDir '.angular\cache'),
        (Join-Path $ClientDir 'node_modules\.cache'),
        (Join-Path $ClientDir 'coverage')
    )
    $resourceRoots = @(
        (Join-Path $RepoRoot 'app\resources'),
        (Get-ConfiguredResourceRoot)
    ) | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') } | Select-Object -Unique
    foreach ($resourceRoot in $resourceRoots) {
        $legacyPaths += @(
            (Join-Path $resourceRoot 'cache'),
            (Join-Path $resourceRoot 'caches'),
            (Join-Path $resourceRoot '.cache'),
            (Join-Path $resourceRoot '__pycache__'),
            (Join-Path $resourceRoot 'models\.cache'),
            (Join-Path $resourceRoot 'models\huggingface\hub-cache'),
            (Join-Path $resourceRoot 'models\huggingface\.cache'),
            (Join-Path $resourceRoot 'models\huggingface\cache'),
            (Join-Path $resourceRoot 'models\torch'),
            (Join-Path $resourceRoot 'models\keras'),
            (Join-Path $resourceRoot 'matplotlib')
        )
    }
    $qaRoots = @(
        (Join-Path $RepoRoot 'assets\QA'),
        (Join-Path $RepoRoot 'app\server\assets\QA')
    ) | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') } | Select-Object -Unique
    foreach ($qaRoot in $qaRoots) {
        if (Test-Path -LiteralPath $qaRoot -PathType Container) {
            $legacyPaths += @(
                Get-ChildItem -LiteralPath $qaRoot -Directory -Recurse -Force -ErrorAction SilentlyContinue |
                    Where-Object {
                        $_.Name -in @('.pytest_cache', '.ruff_cache', '.mypy_cache') -or
                        $_.Name -like 'pytest-cache*'
                    }
            )
        }
    }

    $legacyNames = @(
        '__pycache__', '.uv-cache', '.pytest_cache', '.ruff_cache', '.mypy_cache',
        '.pyright', '.cache', 'cache', 'caches'
    )
    $excludedNames = @('.git', '.venv', 'node_modules', 'dist', 'build', 'release', 'target')
    $skipSubtrees = @(
        $RuntimesDir,
        $VenvDir,
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $ClientDir 'node_modules'),
        (Join-Path $DesktopDir 'node_modules'),
        $DesktopTargetDir,
        $qaRoots,
        (Join-Path $ClientDir '.angular'),
        (Join-Path $RepoRoot 'app\tests\cache'),
        (Join-Path $RepoRoot 'app\server\app\tests\cache'),
        (Join-Path $ServerDir 'runtimes\cache')
    ) | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') } | Select-Object -Unique
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($RepoRoot)
    $found = [Collections.Generic.List[object]]::new()
    $foundPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($legacyPath in @($legacyPaths)) {
        $candidatePath = if ($legacyPath -is [IO.FileSystemInfo]) { $legacyPath.FullName } else { [string]$legacyPath }
        if (-not (Test-Path -LiteralPath $candidatePath)) { continue }
        $legacyItem = Get-Item -LiteralPath $candidatePath -Force -ErrorAction SilentlyContinue
        if ($null -ne $legacyItem -and $foundPaths.Add($legacyItem.FullName)) {
            [void]$found.Add($legacyItem)
        }
    }
    $progressId = Start-LauncherProgress -Activity 'XREPORT: find legacy caches' -Status 'Scanning repository directories'
    try {
        while ($pending.Count -gt 0) {
            $current = $pending.Pop()
            Update-LauncherProgress -Id $progressId -Activity 'XREPORT: find legacy caches' -Status "Scanning $current"
            try {
                $childDirectories = @([IO.Directory]::EnumerateDirectories($current) |
                    Sort-Object @{ Expression = { $_.ToUpperInvariant() }; Descending = $false })
            } catch {
                continue
            }
            foreach ($childPath in $childDirectories) {
                $childName = Split-Path -Leaf $childPath
                $isSkippedSubtree = $false
                foreach ($skipRoot in $skipSubtrees) {
                    if ($childPath.Equals($skipRoot, [StringComparison]::OrdinalIgnoreCase) -or
                        $childPath.StartsWith("$skipRoot\", [StringComparison]::OrdinalIgnoreCase)) {
                        $isSkippedSubtree = $true
                        break
                    }
                }
                if ($isSkippedSubtree -or $childName -in $excludedNames -or
                    $childPath.StartsWith((Join-Path $RepoRoot 'app\resources') + '\', [StringComparison]::OrdinalIgnoreCase)) {
                    continue
                }
                $isPytestCache = $childName -match '^pytest[-_](cache|tmp|integration|e2e|release|full|runtime|settings)'
                if ($childName -in $legacyNames -or $isPytestCache) {
                    $legacyItem = Get-Item -LiteralPath $childPath -Force -ErrorAction SilentlyContinue
                    if ($null -ne $legacyItem -and $foundPaths.Add($legacyItem.FullName)) {
                        [void]$found.Add($legacyItem)
                    }
                    continue
                }
                $pending.Push([string]$childPath)
            }
        }
        @($found |
            Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    }
    finally {
        Complete-LauncherProgress -Id $progressId
    }
}

function Remove-PythonCaches {
    $caches = @(Get-LegacyCacheDirectories | Where-Object { $_.Name -eq '__pycache__' })
    $results = @($caches | ForEach-Object { Remove-LauncherPath -Path $_.FullName -Activity "XREPORT: remove $($_.FullName)" })
    $removed = [int](($results | Measure-Object -Property Removed -Sum).Sum)
    $skipped = [int](($results | Measure-Object -Property Skipped -Sum).Sum)
    if ($skipped -gt 0) {
        Write-Warn "Removed $removed Python cache item(s); skipped $skipped protected item(s)."
    } else {
        Write-Ok "Removed $removed Python cache item(s)"
    }
}

function Clear-ApplicationCache {
    if (-not (Confirm-DestructiveAction 'clear application caches')) { return }
    $targets = @(
        $RuntimeCacheDir
    )
    $legacyCaches = @(Get-LegacyCacheDirectories)
    $allTargets = @(
        $targets + @($legacyCaches | ForEach-Object { $_.FullName }) |
            ForEach-Object { [IO.Path]::GetFullPath($_) } |
            Select-Object -Unique
    )
    $results = [Collections.Generic.List[object]]::new()
    $progressId = Start-LauncherProgress -Activity 'XREPORT: clear application cache' -Status "0 of $($allTargets.Count) paths"
    try {
        for ($index = 0; $index -lt $allTargets.Count; $index++) {
            $target = $allTargets[$index]
            Update-LauncherProgress -Id $progressId -Activity 'XREPORT: clear application cache' -Status "$($index + 1) of $($allTargets.Count): $target" -PercentComplete ([int](($index + 1) * 100 / [Math]::Max(1, $allTargets.Count)))
            $preserveNames = if ([IO.Path]::GetFullPath($target).TrimEnd('\').Equals([IO.Path]::GetFullPath($RuntimeCacheDir).TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) { @('.gitkeep') } else { @() }
            [void]$results.Add((Remove-LauncherPath -Path $target -PreserveNames $preserveNames -Activity "XREPORT: remove $target"))
        }
    }
    finally {
        Complete-LauncherProgress -Id $progressId
    }

    $removed = [int](($results | Measure-Object -Property Removed -Sum).Sum)
    $skipped = [int](($results | Measure-Object -Property Skipped -Sum).Sum)
    if ($skipped -gt 0) {
        Write-Warn "Application cache cleanup completed: removed $removed item(s); skipped $skipped protected item(s)."
    } else {
        Write-Ok "Application caches cleared: removed $removed item(s)."
    }
    Initialize-Environment
}

function Uninstall-Application {
    if (-not (Confirm-DestructiveAction 'remove application runtimes, dependencies, and build outputs')) { return }
    $targets = @(
        $RuntimesDir,
        $VenvDir,
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $ClientDir 'node_modules'),
        (Join-Path $ClientDir '.angular'),
        (Join-Path $ClientDir 'dist')
    )
    $progressId = Start-LauncherProgress -Activity 'XREPORT: uninstall application' -Status "0 of $($targets.Count) paths"
    try {
        for ($index = 0; $index -lt $targets.Count; $index++) {
            $target = $targets[$index]
            Update-LauncherProgress -Id $progressId -Activity 'XREPORT: uninstall application' -Status "$($index + 1) of $($targets.Count): $target" -PercentComplete ([int](($index + 1) * 100 / [Math]::Max(1, $targets.Count)))
            Remove-LauncherPath -Path $target -Activity "XREPORT: remove $target" | Out-Null
        }
    }
    finally {
        Complete-LauncherProgress -Id $progressId
    }
    Remove-PythonCaches
    foreach ($legacyCache in @(Get-LegacyCacheDirectories)) {
        Remove-LauncherPath -Path $legacyCache.FullName -Activity "XREPORT: remove $($legacyCache.FullName)" | Out-Null
    }
    Write-Ok 'Application runtimes, dependencies, and build outputs removed. Dependency lockfiles and user data were preserved.'
}

function Write-MenuRule {
    param([ConsoleColor]$Color = [ConsoleColor]::DarkCyan)
    Write-Host ('  ' + ('-' * 68)) -ForegroundColor $Color
}

function Write-MenuItem {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Entry,
        [Parameter(Mandatory = $true)][int]$NumberWidth,
        [Parameter(Mandatory = $true)][int]$LabelWidth
    )

    $color = if ($Entry.Destructive) { [ConsoleColor]::Yellow } elseif ($Entry.Key -in @('Exit', 'Back')) { [ConsoleColor]::DarkGray } else { [ConsoleColor]::White }
    Write-Host ("  {0,$NumberWidth}. {1,-$LabelWidth}  {2}" -f $Entry.Number, $Entry.Label, $Entry.Description) -ForegroundColor $color
}

function Wait-ForMenu {
    Clear-LauncherProgress
    if (-not $script:LauncherInteractive) { return }
    Write-Host ''
    Write-Host '  Press any key to return to the menu...' -ForegroundColor DarkGray
    try { [void][Console]::ReadKey($true) } catch { }
}

function Get-LauncherMenuEntries {
    @(
        [pscustomobject]@{ Section = 'APPLICATION'; Key = 'Launch'; Label = 'Launch application'; Description = 'Start local services'; Destructive = $false }
        [pscustomobject]@{ Section = 'APPLICATION'; Key = 'KillProcesses'; Label = 'Kill app processes'; Description = 'Stop XREPORT services and desktop shells'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Key = 'Install'; Label = 'Install / update dependencies'; Description = 'Sync runtimes and packages'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Key = 'Rebuild'; Label = 'Rebuild frontend'; Description = 'Build client without launching services'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Key = 'Database'; Label = 'Initialize database'; Description = 'Prepare local data store'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Key = 'Tests'; Label = 'Run test suite'; Description = 'Execute project checks'; Destructive = $false }
        [pscustomobject]@{ Section = 'SOURCE CONTROL'; Key = 'Update'; Label = 'Update application'; Description = 'Pull origin/main (clean main branch required)'; Destructive = $false }
        [pscustomobject]@{ Section = 'BUILD & DISTRIBUTION'; Key = 'CreateRelease'; Label = 'Create release artifacts'; Description = 'Build selected desktop packages'; Destructive = $false }
        [pscustomobject]@{ Section = 'BUILD & DISTRIBUTION'; Key = 'RemoveRelease'; Label = 'Remove release artifacts'; Description = 'Delete selected desktop packages'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Key = 'Logs'; Label = 'Remove logs'; Description = 'Delete application logs'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Key = 'Cache'; Label = 'Clear cache'; Description = 'Remove temporary caches'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Key = 'Checkpoints'; Label = 'Remove checkpoints'; Description = 'Delete saved model checkpoints'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Key = 'AllData'; Label = 'Remove all data'; Description = 'Delete local database and user-generated data'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Key = 'Uninstall'; Label = 'Uninstall application'; Description = 'Remove generated files'; Destructive = $true }
        [pscustomobject]@{ Section = 'EXIT'; Key = 'Exit'; Label = 'Exit'; Description = 'Close launcher'; Destructive = $false }
    )
}

function Show-Menu {
    Clear-LauncherProgress
    if ($script:LauncherInteractive) { try { Clear-Host } catch { } }
    $entries = @(Get-LauncherMenuEntries)
    for ($index = 0; $index -lt $entries.Count; $index++) {
        $entries[$index] = [pscustomobject]@{
            Number = $index + 1
            Section = $entries[$index].Section
            Key = $entries[$index].Key
            Label = $entries[$index].Label
            Description = $entries[$index].Description
            Destructive = $entries[$index].Destructive
        }
    }
    $numberWidth = ([string]$entries.Count).Length
    $labelWidth = ($entries | ForEach-Object { $_.Label.Length } | Measure-Object -Maximum).Maximum
    Write-Host ''
    Write-Host '  XREPORT' -ForegroundColor Cyan -NoNewline
    Write-Host '  /  RADIOLOGICAL REPORTS' -ForegroundColor White
    Write-Host '  Local workspace console' -ForegroundColor DarkGray
    Write-MenuRule
    $lastSection = $null
    foreach ($entry in $entries) {
        if ($entry.Section -ne $lastSection) {
            if ($null -ne $lastSection) { Write-Host '' }
            Write-Host ("  {0}" -f $entry.Section) -ForegroundColor DarkCyan
            $lastSection = $entry.Section
        }
        Write-MenuItem -Entry $entry -NumberWidth $numberWidth -LabelWidth $labelWidth
    }
    Write-MenuRule -Color DarkGray
    Write-Host ''
    return $entries
}

if ($Launch -and $Action) {
    throw 'Use either -Launch or -Action, not both.'
}

if ($Launch) {
    Invoke-TrackedLauncherAction -Name 'launch application' -Operation { Invoke-Launch }
    exit 0
}

if ($Action) {
    Invoke-TrackedLauncherAction -Name "action $Action" -Operation {
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
            'RemoveCheckpoints' { Remove-Checkpoints }
            'RemoveAllData' { Remove-AllData }
            'Uninstall' { Uninstall-Application }
            'KillProcesses' { Stop-XReportProcesses }
            'Update' { Invoke-Update }
        }
    }
    exit 0
}

while ($true) {
    $entries = @(Show-Menu)
    $maxOption = $entries.Count
    if (-not $script:LauncherInteractive) { break }
    Clear-LauncherProgress
    $selection = ([string](Read-Host "  Select an option (1-$maxOption)")).Trim()
    $selectedNumber = 0
    if (-not [int]::TryParse($selection, [ref]$selectedNumber) -or $selectedNumber -lt 1 -or $selectedNumber -gt $maxOption) {
        Write-Warn "Invalid option. Select a number from 1 through $maxOption."
        Wait-ForMenu
        continue
    }
    $entry = $entries[$selectedNumber - 1]
    if ($entry.Key -eq 'Exit') {
        Clear-LauncherProgress
        break
    }

    try {
        Invoke-TrackedLauncherAction -Name "menu option $($entry.Number)" -Operation {
            switch ($entry.Key) {
                'Launch' { Invoke-Launch; exit 0 }
                'KillProcesses' { Stop-XReportProcesses }
                'Install' { Invoke-InstallOrUpdate }
                'Rebuild' { Invoke-RebuildFrontend }
                'Database' { Invoke-InitializeDatabase }
                'Tests' { Invoke-TestSuite }
                'Update' { Invoke-Update }
                'CreateRelease' { Invoke-CreateDesktopArtifactsMenu }
                'RemoveRelease' { Invoke-RemoveDesktopArtifactsMenu }
                'Logs' { Remove-Logs }
                'Cache' { Clear-ApplicationCache }
                'Checkpoints' { Remove-Checkpoints }
                'AllData' { Remove-AllData }
                'Uninstall' { Uninstall-Application }
            }
        }
    } catch {
        Write-Fatal $_.Exception.Message
        Clear-LauncherProgress
    }
    if (-not $script:LauncherInteractive) { exit 0 }
    Wait-ForMenu
}
