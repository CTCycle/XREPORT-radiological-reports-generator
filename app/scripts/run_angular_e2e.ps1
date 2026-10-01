$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $projectRoot 'app\server\.venv\Scripts\python.exe'
$pytestConfig = Join-Path $projectRoot 'app\server\pyproject.toml'
$testFile = Join-Path $projectRoot 'app\tests\e2e\test_angular_ui.py'
$runtimeCacheDir = Join-Path $projectRoot 'runtimes\cache'
$pytestCacheDir = Join-Path $runtimeCacheDir 'pytest'
$pytestBaseTemp = Join-Path $runtimeCacheDir "pytest-tmp\angular-e2e-$PID"
$playwrightBrowsersPath = Join-Path $runtimeCacheDir 'playwright-browsers'
$pythonCacheDir = Join-Path $runtimeCacheDir 'python'
$matplotlibCacheDir = Join-Path $runtimeCacheDir 'matplotlib'

New-Item -ItemType Directory -Force -Path @(
    $runtimeCacheDir,
    $pytestCacheDir,
    $pytestBaseTemp,
    $playwrightBrowsersPath,
    $pythonCacheDir,
    $matplotlibCacheDir
) | Out-Null
$env:XREPORT_CACHE_ROOT = $runtimeCacheDir
$env:XDG_CACHE_HOME = $runtimeCacheDir
$env:PYTEST_CACHE_DIR = $pytestCacheDir
$env:PYTHONPYCACHEPREFIX = $pythonCacheDir
$env:PLAYWRIGHT_BROWSERS_PATH = $playwrightBrowsersPath
$env:MPLCONFIGDIR = $matplotlibCacheDir
Remove-Item Env:HF_CACHE_DIR, Env:TRANSFORMERS_CACHE -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $python)) {
    throw "Backend virtualenv Python not found: $python"
}

Write-Host '[START] XREPORT Angular E2E tests' -ForegroundColor Cyan
& $python -m pytest -c $pytestConfig $testFile -q --tb=short --basetemp $pytestBaseTemp -o "cache_dir=$pytestCacheDir"
$exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
if ($exitCode -eq 0) {
    Write-Host '[DONE] XREPORT Angular E2E tests passed.' -ForegroundColor Green
}
else {
    Write-Host "[FAIL] XREPORT Angular E2E tests exited with code $exitCode." -ForegroundColor Red
}
exit $exitCode
