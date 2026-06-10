[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$appDir = Join-Path $repoRoot "app"
$pathsToRemove = @(
  (Join-Path $appDir "src-tauri"),
  (Join-Path $repoRoot "release\windows")
)

foreach ($path in $pathsToRemove) {
  if (Test-Path $path) {
    Remove-Item -Recurse -Force $path
    Write-Host "[OK] Removed: $path"
  } else {
    Write-Host "[INFO] Not found: $path"
  }
}

Write-Host "[DONE] Build cleanup complete."

