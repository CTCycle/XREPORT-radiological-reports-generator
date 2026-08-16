$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $projectRoot 'app\server\.venv\Scripts\python.exe'
$testFile = Join-Path $projectRoot 'app\tests\e2e\test_angular_ui.py'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Backend virtualenv Python not found: $python"
}

& $python -m pytest $testFile -q --tb=short
exit $LASTEXITCODE
