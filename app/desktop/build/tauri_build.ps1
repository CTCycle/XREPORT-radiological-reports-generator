[CmdletBinding()]
param(
    [ValidateSet('Cpu', 'Cuda', 'All')]
    [string]$DesktopRuntime = 'Cpu',
    [ValidateSet('Portable', 'Msi', 'All')]
    [string]$DesktopTarget = 'All',
    [string]$Version,
    [switch]$OfflineWebView2,
    [switch]$Force,
    [switch]$AllowDirtyTree
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$launcher = Join-Path $repoRoot 'start_on_windows.ps1'
if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = (Select-String -LiteralPath (Join-Path $repoRoot 'app\server\pyproject.toml') -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches.Groups[1].Value
    if ([string]::IsNullOrWhiteSpace($Version)) { throw 'Could not read the canonical version from app/server/pyproject.toml.' }
}
$launcherArgs = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $launcher,
    '-Action', 'BuildDesktopRelease',
    '-DesktopRuntime', $DesktopRuntime,
    '-DesktopTarget', $DesktopTarget,
    '-Version', $Version
)
if ($OfflineWebView2) { $launcherArgs += '-OfflineWebView2' }
if ($Force) { $launcherArgs += '-Force' }
if ($AllowDirtyTree) { $launcherArgs += '-AllowDirtyTree' }

& powershell.exe @launcherArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
