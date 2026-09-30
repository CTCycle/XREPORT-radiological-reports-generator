[CmdletBinding()]
# Expanded native acceptance driver: scenario groups below are explicit so
# route-only packaged smoke remains separate from complete S40 evidence.
param(
    [string]$WindowTitlePattern = 'XREPORT*',
    [string]$Output = (Join-Path (Get-Location) 'native-webview-validation.json'),
    [string[]]$Routes = @('Inference', 'Dataset', 'Training', 'Reports', 'Settings', 'Help'),
    [string[]]$Scenario = @('Routes'),
    [int]$WaitMilliseconds = 300,
    [int]$TimeoutSeconds = 20,
    [int]$BackendPort = 5003,
    [int]$UiPort = 8003,
    [int]$BackendProcessId = 0,
    [string]$BackendRestartExecutable,
    [string[]]$BackendRestartArgumentList = @(),
    [string]$BackendRestartWorkingDirectory,
    [string]$SecondInstanceExecutable,
    [switch]$CloseAtEnd,
    [switch]$RequirePortCleanup
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

if (-not ('XReportNativeKeyboard' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class XReportNativeKeyboard {
    private const uint KeyUp = 0x0002;
    [DllImport("user32.dll")]
    private static extern void keybd_event(byte virtualKey, byte scanCode, uint flags, UIntPtr extraInfo);
    public static void Press(ushort virtualKey, ushort modifier) {
        if (modifier != 0) keybd_event((byte)modifier, 0, 0, UIntPtr.Zero);
        keybd_event((byte)virtualKey, 0, 0, UIntPtr.Zero);
        keybd_event((byte)virtualKey, 0, KeyUp, UIntPtr.Zero);
        if (modifier != 0) keybd_event((byte)modifier, 0, KeyUp, UIntPtr.Zero);
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

function Find-NamedElement {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window,
        [Parameter(Mandatory = $true)][string[]]$Names
    )
    foreach ($name in $Names) {
        $condition = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::NameProperty,
            $name
        )
        $control = $Window.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants,
            $condition
        )
        if ($null -ne $control) { return $control }
    }
    return $null
}

function Find-NamedTextElement {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window,
        [Parameter(Mandatory = $true)][string[]]$Names
    )
    foreach ($name in $Names) {
        $nameCondition = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::NameProperty,
            $name
        )
        $textCondition = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::Text
        )
        $conditions = [System.Windows.Automation.Condition[]]@($nameCondition, $textCondition)
        $condition = [System.Windows.Automation.AndCondition]::new($conditions)
        $control = $Window.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants,
            $condition
        )
        if ($null -ne $control) { return $control }
    }
    return $null
}

function Get-AccessibleNames {
    param([Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window)
    $all = $Window.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition
    )
    $names = @()
    for ($index = 0; $index -lt $all.Count; $index++) {
        $name = [string]$all[$index].Current.Name
        if (-not [string]::IsNullOrWhiteSpace($name)) { $names += $name }
    }
    return @($names | Select-Object -Unique)
}

function Get-FocusedName {
    param([Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window)
    try {
        $focused = [System.Windows.Automation.AutomationElement]::FocusedElement
        if ($null -ne $focused -and -not [string]::IsNullOrWhiteSpace($focused.Current.Name)) {
            return [string]$focused.Current.Name
        }
        $focusCondition = [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::HasKeyboardFocusProperty,
            $true
        )
        $focusedDescendant = $Window.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants,
            $focusCondition
        )
        if ($null -ne $focusedDescendant) { return [string]$focusedDescendant.Current.Name }
    }
    catch { return '' }
    return ''
}

function Wait-ForNamedElement {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window,
        [Parameter(Mandatory = $true)][string[]]$Names,
        [int]$Seconds = $TimeoutSeconds
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    do {
        $control = Find-NamedElement -Window $Window -Names $Names
        if ($null -ne $control) { return $control }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    return $null
}

function Wait-ForNameToDisappear {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window,
        [Parameter(Mandatory = $true)][string[]]$Names,
        [int]$Seconds = $TimeoutSeconds
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    do {
        if ($null -eq (Find-NamedElement -Window $Window -Names $Names)) { return $true }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Send-NativeKey {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window,
        [Parameter(Mandatory = $true)][int]$VirtualKey,
        [int]$Modifier = 0
    )
    [XReportNativeWindowCapture]::SetForegroundWindow($Window.Current.NativeWindowHandle) | Out-Null
    [XReportNativeKeyboard]::Press([uint16]$VirtualKey, [uint16]$Modifier)
}

function Get-RouteHeadingNames {
    param([Parameter(Mandatory = $true)][string]$Route)
    switch ($Route) {
        'Inference' { return @('Turn a radiograph into a draft report') }
        # The dataset page is intentionally section-led and does not render a
        # page-level h1. Use its first stable section marker instead.
        'Dataset' { return @('Data Source', 'Dataset Processing') }
        'Training' { return @('XREPORT Transformer') }
        'Reports' { return @('Reports') }
        'Settings' { return @('Settings') }
        default { return @($Route) }
    }
}

function Wait-ForRouteHeading {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window,
        [Parameter(Mandatory = $true)][string]$Route,
        [int]$Seconds = $TimeoutSeconds
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    do {
        $heading = Find-NamedTextElement -Window $Window -Names (Get-RouteHeadingNames -Route $Route)
        if ($null -ne $heading) { return $heading }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    return $null
}

function Invoke-Route {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Window,
        [Parameter(Mandatory = $true)][string]$Route,
        [Parameter(Mandatory = $true)][string]$ScreenshotRoot,
        [Parameter(Mandatory = $true)][System.Collections.IDictionary]$Result
    )
    $routeResult = [ordered]@{
        route = $Route
        found = $false
        invoked = $false
        invoke_method = $null
        heading_found = $false
        screenshot = $null
        error = $null
    }
    try {
        $control = Find-RouteControl -Window $Window -Route $Route
        if ($null -eq $control) { throw "Route control '$Route' was not exposed by native UI Automation." }
        $routeResult.found = $true
        $routeResult.invoke_method = Invoke-NativeControl -Control $control -Window $Window
        $routeResult.invoked = $true
        Start-Sleep -Milliseconds $WaitMilliseconds
        if ($Route -eq 'Help') {
            if ($null -eq (Wait-ForNamedElement -Window $Window -Names @('Tips & Tricks'))) {
                throw 'Help route did not expose the Tips & Tricks dialog.'
            }
        }
        else {
            $heading = $null
            $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
            do {
                $heading = Find-NamedTextElement -Window $Window -Names (Get-RouteHeadingNames -Route $Route)
                if ($null -ne $heading) { break }
                Start-Sleep -Milliseconds 250
            } while ([DateTime]::UtcNow -lt $deadline)
            if ($null -eq $heading) { throw "Route '$Route' did not expose its expected visible route marker." }
            $routeResult.heading_found = $true
        }
        $safeRoute = $Route.ToLowerInvariant() -replace '[^a-z0-9]+', '-'
        $screenshot = Join-Path $ScreenshotRoot "$safeRoute.png"
        Save-NativeScreenshot -Window $Window -Path $screenshot
        $routeResult.screenshot = [IO.Path]::GetFileName($screenshot)
        $Result.screenshots += [IO.Path]::GetFileName($screenshot)
        if ($Route -eq 'Help') {
            Send-NativeKey -Window $Window -VirtualKey 0x1B
            if (-not (Wait-ForNameToDisappear -Window $Window -Names @('Tips & Tricks', 'Close Tips and Tricks'))) {
                throw 'Help dialog did not close after Escape.'
            }
        }
    }
    catch {
        $routeResult.error = $_.Exception.Message
    }
    $Result.routes += $routeResult
    return $routeResult
}

function Set-ScenarioResult {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Status,
        [System.Collections.IDictionary]$Details = @{}
    )
    $entry = [ordered]@{ id = $Name; status = $Status }
    foreach ($key in $Details.Keys) { $entry[$key] = $Details[$key] }
    $result.scenarios[$Name] = $entry
}

function Test-ScenarioRequested {
    param([Parameter(Mandatory = $true)][string]$Name)
    return $requestedScenarios -contains $Name
}

function Get-BackendListener {
    if ($null -eq (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) { return $null }
    $connection = Get-NetTCPConnection -State Listen -LocalPort $BackendPort -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $connection) { return $null }
    $process = Get-Process -Id ([int]$connection.OwningProcess) -ErrorAction SilentlyContinue
    if ($null -eq $process) { return $null }
    $metadata = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $($process.Id)" -ErrorAction SilentlyContinue
    return [pscustomobject]@{
        Process = $process
        CommandLine = [string]$metadata.CommandLine
        ExecutablePath = [string]$metadata.ExecutablePath
    }
}

function Wait-ForBackendHealth {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$BackendPort/api/health" -TimeoutSec 2
            if ($response.StatusCode -eq 200) { return $true }
        }
        catch { }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Invoke-SecondInstanceCheck {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$PrimaryProcess)
    $executable = $SecondInstanceExecutable
    if ([string]::IsNullOrWhiteSpace($executable)) {
        $metadata = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $($PrimaryProcess.Id)" -ErrorAction SilentlyContinue
        $executable = [string]$metadata.ExecutablePath
    }
    if ([string]::IsNullOrWhiteSpace($executable) -or -not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        return [pscustomobject]@{ Status = 'UNRUN'; Error = 'A second-instance executable path was not available.' }
    }
    $second = Start-Process -FilePath $executable -PassThru
    $dialog = $null
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $dialog = Get-Process | Where-Object { $_.MainWindowTitle -eq 'XREPORT is already running' } | Select-Object -First 1
        if ($null -ne $dialog) { break }
        if ($second.HasExited) { break }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    if ($null -eq $dialog) {
        if (-not $second.HasExited) { Stop-Process -Id $second.Id -Force -ErrorAction SilentlyContinue }
        return [pscustomobject]@{ Status = 'FAIL'; Error = 'Second instance did not expose the single-instance message.' }
    }
    $dialogElement = [System.Windows.Automation.AutomationElement]::FromHandle($dialog.MainWindowHandle)
    $ok = Find-NamedElement -Window $dialogElement -Names @('OK')
    if ($null -eq $ok) { throw 'Single-instance message did not expose an OK control.' }
    Invoke-NativeControl -Control $ok -Window $dialogElement | Out-Null
    $second.WaitForExit(5000)
    if (-not $second.HasExited) { return [pscustomobject]@{ Status = 'FAIL'; Error = 'Second instance remained alive after the policy dialog was dismissed.' } }
    return [pscustomobject]@{ Status = 'PASS'; Error = $null }
}

$allScenarioNames = @('startup', 'routes', 'history', 'keyboard', 'modal', 'slow_readiness', 'backend_stop', 'backend_retry', 'second_instance', 'cleanup')
$allowedScenarioNames = @('All', 'Routes', 'History', 'Keyboard', 'Modal', 'SlowReadiness', 'BackendStop', 'BackendRetry', 'SecondInstance', 'Cleanup')
$scenarioValues = @($Scenario | ForEach-Object { $_ -split ',' | ForEach-Object { $_.Trim() } } | Where-Object { $_ })
$invalidScenarioNames = @($scenarioValues | Where-Object { $_ -notin $allowedScenarioNames })
if ($invalidScenarioNames.Count -gt 0) {
    throw "Unsupported scenario name(s): $($invalidScenarioNames -join ', '). Allowed values: $($allowedScenarioNames -join ', ')."
}
$requestedScenarios = if ($scenarioValues -contains 'All') { @($allScenarioNames) } else {
    $scenarioValues | ForEach-Object {
        switch ($_) {
            'Routes' { 'routes' }
            'History' { 'history' }
            'Keyboard' { 'keyboard' }
            'Modal' { 'modal' }
            'SlowReadiness' { 'slow_readiness' }
            'BackendStop' { 'backend_stop' }
            'BackendRetry' { 'backend_retry' }
            'SecondInstance' { 'second_instance' }
            'Cleanup' { 'cleanup' }
        }
    }
}
if ($requestedScenarios.Count -eq 0) { throw 'At least one validation scenario must be selected.' }

$outputPath = [IO.Path]::GetFullPath($Output)
$screenshotRoot = Join-Path (Split-Path -Parent $outputPath) 'native-screenshots'
New-Item -ItemType Directory -Path $screenshotRoot -Force | Out-Null
$native = Find-NativeWindow
$window = $native.Element
$result = [ordered]@{
    format = 2
    observed_utc = [DateTime]::UtcNow.ToString('o')
    process_id = [int]$native.Process.Id
    window_title = $native.Process.MainWindowTitle
    native_window_handle = [int64]$native.Process.MainWindowHandle
    requested_scenarios = @($requestedScenarios)
    screenshots = @()
    routes = @()
    scenarios = [ordered]@{}
    accessible_names = @()
    passed = $false
}

foreach ($scenarioName in $allScenarioNames) {
    $result.scenarios[$scenarioName] = [ordered]@{ id = $scenarioName; status = 'UNRUN' }
}

$startupScreenshot = Join-Path $screenshotRoot 'startup.png'
Save-NativeScreenshot -Window $window -Path $startupScreenshot
$result.screenshots += [IO.Path]::GetFileName($startupScreenshot)
$initialNames = @(Get-AccessibleNames -Window $window)
$result.accessible_names = $initialNames

# Finding the ready navigation is also the startup assertion for every live
# route run. The dedicated slow-readiness scenario must attach before it appears.
$inferenceControl = Wait-ForNamedElement -Window $window -Names @('Inference')
if ($null -ne $inferenceControl) {
    Set-ScenarioResult -Name 'startup' -Status 'PASS' -Details @{ ready_route = 'Inference' }
}
else {
    Set-ScenarioResult -Name 'startup' -Status 'FAIL' -Details @{ error = 'Native window did not expose the ready workspace.' }
}

if (Test-ScenarioRequested -Name 'routes') {
    foreach ($route in $Routes) { Invoke-Route -Window $window -Route $route -ScreenshotRoot $screenshotRoot -Result $result | Out-Null }
    $failedRoutes = @($result.routes | Where-Object { -not $_.invoked -or ($_.route -ne 'Help' -and -not $_.heading_found) -or [string]::IsNullOrWhiteSpace($_.screenshot) })
    if ($failedRoutes.Count -eq 0) {
        Set-ScenarioResult -Name 'routes' -Status 'PASS' -Details @{ route_count = $Routes.Count }
    }
    else {
        Set-ScenarioResult -Name 'routes' -Status 'FAIL' -Details @{ failed_routes = @($failedRoutes | ForEach-Object { $_.route }) }
    }
}

if (Test-ScenarioRequested -Name 'history') {
    $historyDetails = @{}
    try {
        Invoke-Route -Window $window -Route 'Inference' -ScreenshotRoot $screenshotRoot -Result $result | Out-Null
        Invoke-Route -Window $window -Route 'Dataset' -ScreenshotRoot $screenshotRoot -Result $result | Out-Null
        Send-NativeKey -Window $window -VirtualKey 0x74
        $refreshOk = $null -ne (Wait-ForRouteHeading -Window $window -Route 'Dataset')
        Send-NativeKey -Window $window -VirtualKey 0x25 -Modifier 0x12
        $backOk = $null -ne (Wait-ForRouteHeading -Window $window -Route 'Inference')
        Send-NativeKey -Window $window -VirtualKey 0x27 -Modifier 0x12
        $forwardOk = $null -ne (Wait-ForRouteHeading -Window $window -Route 'Dataset')
        $historyDetails.refresh = $refreshOk
        $historyDetails.back = $backOk
        $historyDetails.forward = $forwardOk
        $historyStatus = if ($refreshOk -and $backOk -and $forwardOk) { 'PASS' } else { 'FAIL' }
        if ($historyStatus -eq 'FAIL') { $historyDetails.error = 'Refresh/history navigation did not restore the expected route headings.' }
        Set-ScenarioResult -Name 'history' -Status $historyStatus -Details $historyDetails
    }
    catch {
        $historyDetails.error = $_.Exception.Message
        Set-ScenarioResult -Name 'history' -Status 'FAIL' -Details $historyDetails
    }
}

if (Test-ScenarioRequested -Name 'keyboard') {
    try {
        Invoke-Route -Window $window -Route 'Inference' -ScreenshotRoot $screenshotRoot -Result $result | Out-Null
        $focusedNames = @()
        for ($index = 0; $index -lt 18; $index++) {
            Send-NativeKey -Window $window -VirtualKey 0x09
            Start-Sleep -Milliseconds 80
            $focused = Get-FocusedName -Window $window
            if (-not [string]::IsNullOrWhiteSpace($focused)) { $focusedNames += $focused }
        }
        $navigationNames = @('Inference', 'Reports', 'Dataset', 'Training', 'Help and tips', 'Settings')
        $navigationFocusCount = @($focusedNames | Where-Object { $_ -in $navigationNames } | Select-Object -Unique).Count
        if ($navigationFocusCount -ge 4) {
            Set-ScenarioResult -Name 'keyboard' -Status 'PASS' -Details @{ focused_names = @($focusedNames | Select-Object -Unique); navigation_focus_count = $navigationFocusCount }
        }
        else {
            Set-ScenarioResult -Name 'keyboard' -Status 'FAIL' -Details @{ focused_names = @($focusedNames | Select-Object -Unique); navigation_focus_count = $navigationFocusCount; error = 'Keyboard traversal did not expose at least four primary navigation controls.' }
        }
    }
    catch {
        Set-ScenarioResult -Name 'keyboard' -Status 'FAIL' -Details @{ error = $_.Exception.Message }
    }
}

if (Test-ScenarioRequested -Name 'modal') {
    try {
        Invoke-Route -Window $window -Route 'Inference' -ScreenshotRoot $screenshotRoot -Result $result | Out-Null
        $help = Find-RouteControl -Window $window -Route 'Help'
        if ($null -eq $help) { throw 'Help control was not available for modal validation.' }
        Invoke-NativeControl -Control $help -Window $window | Out-Null
        if ($null -eq (Wait-ForNamedElement -Window $window -Names @('Tips & Tricks'))) { throw 'Tips & Tricks dialog did not open.' }
        $modalFocusNames = @()
        for ($index = 0; $index -lt 12; $index++) {
            Send-NativeKey -Window $window -VirtualKey 0x09
            Start-Sleep -Milliseconds 80
            $focused = Get-FocusedName -Window $window
            if (-not [string]::IsNullOrWhiteSpace($focused)) { $modalFocusNames += $focused }
        }
        $outsideModalNames = @($modalFocusNames | Where-Object { $_ -in @('Inference', 'Reports', 'Dataset', 'Training', 'Help and tips', 'Settings') } | Select-Object -Unique)
        Send-NativeKey -Window $window -VirtualKey 0x1B
        $closed = Wait-ForNameToDisappear -Window $window -Names @('Tips & Tricks', 'Close Tips and Tricks')
        $restoredFocus = (Get-FocusedName -Window $window) -eq 'Help and tips'
        if ($outsideModalNames.Count -eq 0 -and $closed -and $restoredFocus) {
            Set-ScenarioResult -Name 'modal' -Status 'PASS' -Details @{ focused_names = @($modalFocusNames | Select-Object -Unique); focus_restored = $true }
        }
        else {
            Set-ScenarioResult -Name 'modal' -Status 'FAIL' -Details @{ focused_names = @($modalFocusNames | Select-Object -Unique); outside_modal_focus = $outsideModalNames; closed = $closed; focus_restored = $restoredFocus; error = 'Modal focus did not remain trapped and restore to Help and tips.' }
        }
    }
    catch {
        Set-ScenarioResult -Name 'modal' -Status 'FAIL' -Details @{ error = $_.Exception.Message }
    }
}

if (Test-ScenarioRequested -Name 'slow_readiness') {
    $slowNames = @($initialNames | Where-Object { $_ -match '(?i)Preparing XREPORT|still initializing|could not reach the local backend' })
    $sawSlow = @($initialNames | Where-Object { $_ -match '(?i)still initializing' }).Count -gt 0
    $ready = $null -ne (Wait-ForNamedElement -Window $window -Names @('Inference'))
    if ($sawSlow -and $ready) {
        Set-ScenarioResult -Name 'slow_readiness' -Status 'PASS' -Details @{ startup_states = $slowNames; ready = $true }
    }
    else {
        Set-ScenarioResult -Name 'slow_readiness' -Status 'UNRUN' -Details @{ startup_states = $slowNames; ready = $ready; error = 'Run this scenario while a controlled delayed backend is still showing the native startup screen.' }
    }
}

$backendListener = $null
if ((Test-ScenarioRequested -Name 'backend_stop') -or (Test-ScenarioRequested -Name 'backend_retry')) {
    try {
        $backendListener = if ($BackendProcessId -gt 0) {
            $process = Get-Process -Id $BackendProcessId -ErrorAction Stop
            [pscustomobject]@{ Process = $process; CommandLine = ''; ExecutablePath = $process.Path }
        }
        else { Get-BackendListener }
        if ($null -eq $backendListener) { throw "No backend listener was found on port $BackendPort." }
        Stop-Process -Id $backendListener.Process.Id -Force
        Start-Sleep -Milliseconds $WaitMilliseconds
        if (Test-ScenarioRequested -Name 'backend_stop') {
            Invoke-Route -Window $window -Route 'Settings' -ScreenshotRoot $screenshotRoot -Result $result | Out-Null
            $stopNames = @(Get-AccessibleNames -Window $window)
            $errorVisible = @($stopNames | Where-Object { $_ -match '(?i)failed to load|could not|unavailable|backend|connection refused|service' }).Count -gt 0
            if ($errorVisible) {
                Set-ScenarioResult -Name 'backend_stop' -Status 'PASS' -Details @{ backend_process_id = [int]$backendListener.Process.Id; error_state_visible = $true }
            }
            else {
                Set-ScenarioResult -Name 'backend_stop' -Status 'FAIL' -Details @{ backend_process_id = [int]$backendListener.Process.Id; error_state_visible = $false; error = 'Native UI did not expose a backend outage state after the listener was stopped.' }
            }
        }
    }
    catch {
        if (Test-ScenarioRequested -Name 'backend_stop') { Set-ScenarioResult -Name 'backend_stop' -Status 'UNRUN' -Details @{ error = $_.Exception.Message } }
        if (Test-ScenarioRequested -Name 'backend_retry') { Set-ScenarioResult -Name 'backend_retry' -Status 'UNRUN' -Details @{ error = $_.Exception.Message } }
    }
}

if ((Test-ScenarioRequested -Name 'backend_retry') -and $result.scenarios.backend_retry.status -eq 'UNRUN' -and $null -ne $backendListener) {
    try {
        if ([string]::IsNullOrWhiteSpace($BackendRestartExecutable)) { throw 'BackendRestartExecutable is required to run controlled backend recovery.' }
        $workingDirectory = if ([string]::IsNullOrWhiteSpace($BackendRestartWorkingDirectory)) { (Get-Location).Path } else { $BackendRestartWorkingDirectory }
        Start-Process -FilePath $BackendRestartExecutable -ArgumentList $BackendRestartArgumentList -WorkingDirectory $workingDirectory | Out-Null
        $health = Wait-ForBackendHealth
        if (-not $health) { throw "Backend did not recover on port $BackendPort." }
        Send-NativeKey -Window $window -VirtualKey 0x74
        $settingsReady = $null -ne (Wait-ForRouteHeading -Window $window -Route 'Settings')
        if (-not $settingsReady) { throw 'Settings did not recover after backend restart and WebView refresh.' }
        Set-ScenarioResult -Name 'backend_retry' -Status 'PASS' -Details @{ backend_port = $BackendPort; health = $true; settings_ready = $true }
    }
    catch {
        Set-ScenarioResult -Name 'backend_retry' -Status 'FAIL' -Details @{ error = $_.Exception.Message }
    }
}
elseif ((Test-ScenarioRequested -Name 'backend_retry') -and $result.scenarios.backend_retry.status -eq 'UNRUN') {
    Set-ScenarioResult -Name 'backend_retry' -Status 'UNRUN' -Details @{ error = 'Backend stop did not produce a restartable listener state.' }
}

if (Test-ScenarioRequested -Name 'second_instance') {
    try {
        $secondResult = Invoke-SecondInstanceCheck -PrimaryProcess $native.Process
        Set-ScenarioResult -Name 'second_instance' -Status $secondResult.Status -Details @{ error = $secondResult.Error }
    }
    catch {
        Set-ScenarioResult -Name 'second_instance' -Status 'FAIL' -Details @{ error = $_.Exception.Message }
    }
}

if (Test-ScenarioRequested -Name 'cleanup') {
    if (-not $CloseAtEnd) {
        Set-ScenarioResult -Name 'cleanup' -Status 'UNRUN' -Details @{ error = 'Pass -CloseAtEnd to exercise native close and process cleanup.' }
    }
    else {
        try {
            Send-NativeKey -Window $window -VirtualKey 0x73 -Modifier 0x12
            $native.Process.WaitForExit($TimeoutSeconds * 1000)
            $native.Process.Refresh()
            $processExited = $native.Process.HasExited
            $listeners = @()
            if ($null -ne (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) {
                foreach ($port in @($BackendPort, $UiPort)) {
                    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
                        Select-Object -First 1
                    if ($null -ne $listener) { $listeners += [int]$port }
                }
            }
            $portsClean = $listeners.Count -eq 0
            $cleanupPass = $processExited -and (-not $RequirePortCleanup -or $portsClean)
            $cleanupDetails = @{ native_process_exited = $processExited; remaining_listener_ports = $listeners; port_cleanup_required = [bool]$RequirePortCleanup }
            if (-not $cleanupPass) { $cleanupDetails.error = 'Native process or required listeners remained after close.' }
            $cleanupStatus = if ($cleanupPass) { 'PASS' } else { 'FAIL' }
            Set-ScenarioResult -Name 'cleanup' -Status $cleanupStatus -Details $cleanupDetails
        }
        catch {
            Set-ScenarioResult -Name 'cleanup' -Status 'FAIL' -Details @{ error = $_.Exception.Message }
        }
    }
}

try { $result.accessible_names = @(Get-AccessibleNames -Window $window) } catch { }
$requiredScenarios = @(@('startup') + @($requestedScenarios) | Select-Object -Unique)
$result.required_scenarios = $requiredScenarios
$result.passed = @($requiredScenarios | Where-Object { $result.scenarios[$_].status -ne 'PASS' }).Count -eq 0
$result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $outputPath -Encoding utf8
if (-not $result.passed) {
    throw "Native WebView validation failed: $($result | ConvertTo-Json -Compress)"
}
Write-Host "Native WebView validation passed: $outputPath"
