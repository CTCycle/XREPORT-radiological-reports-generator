$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $projectRoot 'app\server\.venv\Scripts\python.exe'
$pytestConfig = Join-Path $projectRoot 'app\server\pyproject.toml'
$testFile = Join-Path $projectRoot 'app\tests\e2e\test_angular_ui.py'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Backend virtualenv Python not found: $python"
}

Write-Host '[START] XREPORT Angular E2E tests' -ForegroundColor Cyan
& $python -m pytest -c $pytestConfig $testFile -q --tb=short
$exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
if ($exitCode -eq 0) {
    Write-Host '[DONE] XREPORT Angular E2E tests passed.' -ForegroundColor Green
}
else {
    Write-Host "[FAIL] XREPORT Angular E2E tests exited with code $exitCode." -ForegroundColor Red
}
exit $exitCode
