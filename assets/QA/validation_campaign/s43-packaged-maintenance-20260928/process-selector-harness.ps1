$ErrorActionPreference = 'Stop'

$evidenceRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$repoRoot = [IO.Path]::GetFullPath((Join-Path $evidenceRoot '..\..\..\..')).TrimEnd('\')
$launcherPath = Join-Path $repoRoot 'start_on_windows.ps1'

$tokens = $null
$parseErrors = $null
$launcherAst = [System.Management.Automation.Language.Parser]::ParseFile(
    $launcherPath,
    [ref]$tokens,
    [ref]$parseErrors
)
if ($parseErrors.Count -gt 0) {
    throw "Launcher parse failed: $($parseErrors[0].Message)"
}
$selectorNode = $launcherAst.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Get-XReportApplicationProcessIds'
}, $true) | Select-Object -First 1
if (-not $selectorNode) {
    throw 'Get-XReportApplicationProcessIds was not found.'
}

$global:RepoRoot = $repoRoot
Invoke-Expression $selectorNode.Extent.Text

$processTable = @(
    [pscustomobject]@{
        ProcessId = 2000000010
        ParentProcessId = 1
        Name = 'XREPORT-v3.1.0-windows-x64-cpu-portable.exe'
        ExecutablePath = (Join-Path $repoRoot 'release\XREPORT-v3.1.0-windows-x64-cpu-portable.exe')
        CommandLine = '"XREPORT-v3.1.0-windows-x64-cpu-portable.exe"'
    }
    [pscustomobject]@{
        ProcessId = 2000000011
        ParentProcessId = 2000000010
        Name = 'XREPORT-backend.exe'
        ExecutablePath = 'C:\Users\Test\AppData\Local\XREPORT\runtime\cpu\3.1.0\hash\backend\XREPORT-backend.exe'
        CommandLine = 'XREPORT-backend.exe --port 0 --variant cpu --version 3.1.0'
    }
    [pscustomobject]@{
        ProcessId = 2000000012
        ParentProcessId = 1
        Name = 'XREPORT-v3.1.0-windows-x64-cuda-portable.exe'
        ExecutablePath = (Join-Path $repoRoot 'release\XREPORT-v3.1.0-windows-x64-cuda-portable.exe')
        CommandLine = '"XREPORT-v3.1.0-windows-x64-cuda-portable.exe"'
    }
    [pscustomobject]@{
        ProcessId = 2000000013
        ParentProcessId = 2000000012
        Name = 'XREPORT-backend.exe'
        ExecutablePath = 'C:\Users\Test\AppData\Local\XREPORT\runtime\cuda\3.1.0\hash\backend\XREPORT-backend.exe'
        CommandLine = 'XREPORT-backend.exe --port 0 --variant cuda --version 3.1.0'
    }
    [pscustomobject]@{
        ProcessId = 2000000014
        ParentProcessId = 1
        Name = 'xreport-desktop.exe'
        ExecutablePath = 'C:\Program Files\XREPORT\xreport-desktop.exe'
        CommandLine = '"C:\Program Files\XREPORT\xreport-desktop.exe"'
    }
    [pscustomobject]@{
        ProcessId = 2000000015
        ParentProcessId = 1
        Name = 'XREPORT-v3.1.0-windows-x64-cpu-portable-helper.exe'
        ExecutablePath = 'C:\Tools\XREPORT-v3.1.0-windows-x64-cpu-portable-helper.exe'
        CommandLine = 'XREPORT-v3.1.0-windows-x64-cpu-portable-helper.exe'
    }
)

$selected = @(Get-XReportApplicationProcessIds -ProcessTable $processTable)
$expected = @(2000000010, 2000000011, 2000000012, 2000000013, 2000000014)
if (($selected -join ',') -ne ($expected -join ',')) {
    throw "Packaged selector mismatch. Expected $($expected -join ','); got $($selected -join ',')."
}

[pscustomobject]@{
    launcher = $launcherPath
    selected_process_ids = $selected
    expected_process_ids = $expected
    excluded_process_ids = @(2000000015)
    portable_cpu_selected = $selected -contains 2000000010
    portable_cuda_selected = $selected -contains 2000000012
    packaged_backend_children_selected = ($selected -contains 2000000011) -and ($selected -contains 2000000013)
    msi_style_desktop_selected = $selected -contains 2000000014
    unrelated_helper_excluded = -not ($selected -contains 2000000015)
    status = 'PASS'
} | ConvertTo-Json -Depth 5
