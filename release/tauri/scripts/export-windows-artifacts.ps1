[CmdletBinding()]
param(
  [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$clientDir = Join-Path $repoRoot "XREPORT\client"
$releaseDir = Join-Path $clientDir "src-tauri\target\release"
$bundleDir = Join-Path $releaseDir "bundle"

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $outputDir = Join-Path $repoRoot "release\windows"
} else {
  $outputDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
}

$installersDir = Join-Path $outputDir "installers"
$portableDir = Join-Path $outputDir "portable"

if (-not (Test-Path $bundleDir)) {
  throw "Bundle directory not found. Run 'npm run tauri:build' first. Missing: $bundleDir"
}

if (Test-Path $outputDir) {
  Remove-Item -Recurse -Force $outputDir
}

New-Item -ItemType Directory -Path $installersDir -Force | Out-Null
New-Item -ItemType Directory -Path $portableDir -Force | Out-Null

$installerArtifacts = @()

$nsisDir = Join-Path $bundleDir "nsis"
if (Test-Path $nsisDir) {
  $nsisFiles = Get-ChildItem -Path $nsisDir -Filter "*.exe" -File
  foreach ($file in $nsisFiles) {
    Copy-Item -Path $file.FullName -Destination $installersDir -Force
    $installerArtifacts += Join-Path $installersDir $file.Name
  }
}

$msiDir = Join-Path $bundleDir "msi"
if (Test-Path $msiDir) {
  $msiFiles = Get-ChildItem -Path $msiDir -Filter "*.msi" -File
  foreach ($file in $msiFiles) {
    Copy-Item -Path $file.FullName -Destination $installersDir -Force
    $installerArtifacts += Join-Path $installersDir $file.Name
  }
}

$portableExeCandidates = Get-ChildItem -Path $releaseDir -Filter "*.exe" -File |
  Where-Object { $_.Name -notmatch "(?i)(setup|installer|uninstall|updater)" }

foreach ($file in $portableExeCandidates) {
  Copy-Item -Path $file.FullName -Destination $portableDir -Force
}

$requiredReleaseEntries = @(
  @{ Name = "backend payload"; Path = (Join-Path $releaseDir "XREPORT") },
  @{ Name = "runtime payload"; Path = (Join-Path $releaseDir "runtimes") },
  @{ Name = "pyproject.toml"; Path = (Join-Path $releaseDir "pyproject.toml") },
  @{ Name = "uv.lock"; Path = (Join-Path $releaseDir "uv.lock") }
)

foreach ($entry in $requiredReleaseEntries) {
  if (-not (Test-Path $entry.Path)) {
    throw "Missing $($entry.Name) in release payload: $($entry.Path). Run release\tauri\build_with_tauri.bat and ensure runtime resources are staged at release root."
  }
}

$portableResourceEntries = @(
  "XREPORT",
  "runtimes",
  "pyproject.toml",
  "uv.lock",
  "resources",
  "_up_"
)

foreach ($entry in $portableResourceEntries) {
  $sourcePath = Join-Path $releaseDir $entry
  if (Test-Path $sourcePath) {
    $destinationPath = Join-Path $portableDir $entry
    Copy-Item -Path $sourcePath -Destination $destinationPath -Recurse -Force
  }
}

$instructions = @"
XREPORT desktop build output

1) Preferred for users:
   Open installers\ and run the setup executable (.exe) or .msi.

2) Portable executable:
   portable\ contains the app .exe and the required runtime resource payload.
   Keep the exported contents together in the same directory.

Generated from:
$bundleDir
"@
Set-Content -Path (Join-Path $outputDir "README.txt") -Value $instructions -Encoding ascii

$requiredPortablePaths = @(
  (Join-Path $portableDir "runtimes\uv\uv.exe"),
  (Join-Path $portableDir "runtimes\python\python.exe"),
  (Join-Path $portableDir "XREPORT"),
  (Join-Path $portableDir "pyproject.toml"),
  (Join-Path $portableDir "uv.lock")
)

foreach ($requiredPath in $requiredPortablePaths) {
  if (-not (Test-Path $requiredPath)) {
    throw "Portable export validation failed; required path missing: $requiredPath"
  }
}

Write-Host "[OK] Exported Windows artifacts to: $outputDir"
Write-Host "[INFO] Installers:"
if ($installerArtifacts.Count -eq 0) {
  Write-Host " - none found"
} else {
  $installerArtifacts | ForEach-Object { Write-Host " - $_" }
}
Write-Host "[INFO] Portable executables:"
$portableFiles = Get-ChildItem -Path $portableDir -Filter "*.exe" -File
if ($portableFiles.Count -eq 0) {
  Write-Host " - none found"
} else {
  $portableFiles | ForEach-Object { Write-Host " - $($_.FullName)" }
}

