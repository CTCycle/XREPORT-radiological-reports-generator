[CmdletBinding()]
param(
    [string]$WindowTitlePattern = 'XREPORT*',
    [string]$Output = (Join-Path (Get-Location) 'native-webview-validation.json'),
    [string[]]$Routes = @('Inference', 'Dataset', 'Training', 'Reports', 'Settings', 'Help'),
    [int]$WaitMilliseconds = 300
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Drawing

if (-not ('XReportNativeWindowCapture' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class XReportNativeWindowCapture {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
}
'@
}

if (-not ('XReportNativeMouse' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class XReportNativeMouse {
    private const uint LeftDown = 0x0002;
    private const uint LeftUp = 0x0004;
    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")]
    private static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extraInfo);
    public static void Click(int x, int y) {
        if (!SetCursorPos(x, y)) throw new InvalidOperationException("SetCursorPos failed.");
        mouse_event(LeftDown, 0, 0, 0, UIntPtr.Zero);
        mouse_event(LeftUp, 0, 0, 0, UIntPtr.Zero);
    }
}
'@
}

function Find-NativeWindow {
    $windows = @(
        Get-Process | Where-Object {
            $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like $WindowTitlePattern
        }
    )
    if ($windows.Count -ne 1) {
        throw "Expected exactly one native XREPORT window matching '$WindowTitlePattern'; found $($windows.Count)."
    }
    $process = $windows[0]
    $element = [System.Windows.Automation.AutomationElement]::FromHandle($process.MainWindowHandle)
    if ($null -eq $element) { throw 'Could not obtain a UI Automation element for the native window.' }
    return [pscustomobject]@{ Process = $process; Element = $element }
}

function Save-NativeScreenshot {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $rect = New-Object XReportNativeWindowCapture+RECT
    if (-not [XReportNativeWindowCapture]::GetWindowRect($Window.Current.NativeWindowHandle, [ref]$rect)) {
        throw 'GetWindowRect failed for the native XREPORT window.'
    }
    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top
    if ($width -le 0 -or $height -le 0) { throw 'Native XREPORT window has invalid bounds.' }
    $bitmap = New-Object System.Drawing.Bitmap($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

function Find-RouteControl {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window,
        [Parameter(Mandatory = $true)][string]$Route
    )
    $candidateNames = if ($Route -eq 'Help') { @('Help', 'Help and tips') } else { @($Route) }
    foreach ($candidateName in $candidateNames) {
        $condition = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::NameProperty,
            $candidateName
        )
        $control = $Window.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants,
            $condition
        )
        if ($null -ne $control) { return $control }
    }
    return $null
}

function Invoke-NativeControl {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Control,
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window
    )
    try {
        $pattern = $Control.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
        $pattern.Invoke()
        return 'invoke_pattern'
    } catch {
        $bounds = $Control.Current.BoundingRectangle
        if ($bounds.Width -le 0 -or $bounds.Height -le 0) {
            throw "Native control is not invokable and has no usable bounds: $($Control.Current.Name)"
        }
        [XReportNativeWindowCapture]::SetForegroundWindow($Window.Current.NativeWindowHandle) | Out-Null
        $x = [int][Math]::Round($bounds.Left + ($bounds.Width / 2))
        $y = [int][Math]::Round($bounds.Top + ($bounds.Height / 2))
        [XReportNativeMouse]::Click($x, $y)
        return 'mouse_click'
    }
}

$outputPath = [IO.Path]::GetFullPath($Output)
$screenshotRoot = Join-Path (Split-Path -Parent $outputPath) 'native-screenshots'
New-Item -ItemType Directory -Path $screenshotRoot -Force | Out-Null
$native = Find-NativeWindow
$window = $native.Element
$result = [ordered]@{
    format = 1
    observed_utc = [DateTime]::UtcNow.ToString('o')
    process_id = [int]$native.Process.Id
    window_title = $native.Process.MainWindowTitle
    native_window_handle = [int64]$native.Process.MainWindowHandle
    screenshots = @()
    routes = @()
    accessible_names = @()
    passed = $false
}

$startupScreenshot = Join-Path $screenshotRoot 'startup.png'
Save-NativeScreenshot -Window $window -Path $startupScreenshot
$result.screenshots += [IO.Path]::GetFileName($startupScreenshot)

foreach ($route in $Routes) {
    $routeResult = [ordered]@{ route = $route; found = $false; invoked = $false; invoke_method = $null; screenshot = $null; error = $null }
    try {
        $control = Find-RouteControl -Window $window -Route $route
        if ($null -eq $control) { throw "Route control '$route' was not exposed by native UI Automation." }
        $routeResult.found = $true
        $routeResult.invoke_method = Invoke-NativeControl -Control $control -Window $window
        $routeResult.invoked = $true
        Start-Sleep -Milliseconds $WaitMilliseconds
        $safeRoute = $route.ToLowerInvariant() -replace '[^a-z0-9]+', '-'
        $screenshot = Join-Path $screenshotRoot "$safeRoute.png"
        Save-NativeScreenshot -Window $window -Path $screenshot
        $routeResult.screenshot = [IO.Path]::GetFileName($screenshot)
        $result.screenshots += [IO.Path]::GetFileName($screenshot)
    } catch {
        $routeResult.error = $_.Exception.Message
    }
    $result.routes += $routeResult
}

$all = $window.FindAll(
    [System.Windows.Automation.TreeScope]::Descendants,
    [System.Windows.Automation.Condition]::TrueCondition
)
$result.accessible_names = @(
    0..($all.Count - 1) |
        ForEach-Object { $all[$_].Current.Name } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Select-Object -Unique
)
$result.passed = @($result.routes | Where-Object { -not $_.invoked }).Count -eq 0
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outputPath -Encoding utf8
if (-not $result.passed) {
    throw "Native WebView validation failed: $($result | ConvertTo-Json -Compress)"
}
Write-Host "Native WebView validation passed: $outputPath"
