[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda')]
    [Parameter(Mandatory = $true)][string]$Variant,
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$SourceCommit,
    [string]$ReleaseRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
if (-not $ReleaseRoot) { $ReleaseRoot = Join-Path $repoRoot 'release' }
if (-not $SourceCommit) { $SourceCommit = (git -C $repoRoot rev-parse HEAD).Trim() }
$python = Join-Path $repoRoot 'runtimes\python\python.exe'
$runtimeVerifier = Join-Path $PSScriptRoot 'verify_runtime_bundle.py'
$prefix = "XREPORT-v$Version-windows-x64-$Variant"
$portable = Join-Path $ReleaseRoot "$prefix-portable.exe"
$msi = Join-Path $ReleaseRoot "$prefix.msi"
$checksum = Join-Path $ReleaseRoot "$prefix.sha256"
$metadataPath = Join-Path $ReleaseRoot "$prefix-build.json"
$runtimeAudit = Join-Path $repoRoot "assets\QA\desktop\runtime-$Variant-$Version.json"
$script:VerificationProgressId = 1

function Update-VerificationProgress {
    param([Parameter(Mandatory = $true)][string]$Status, [Nullable[int]]$PercentComplete)
    $progress = @{ Id = $script:VerificationProgressId; Activity = "XREPORT artifact verification: $Variant"; Status = $Status }
    if ($null -ne $PercentComplete) { $progress.PercentComplete = $PercentComplete }
    Write-Progress @progress
}

function Assert-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Expected desktop artifact is missing: $Path" }
}

function Invoke-Checked {
    param([string]$FilePath, [string[]]$ArgumentList = @(), [string]$WorkingDirectory = $repoRoot)
    $display = "$FilePath " + ($ArgumentList -join ' ')
    Write-Host "[RUN] $display" -ForegroundColor Cyan
    Update-VerificationProgress -Status "Running $display"
    Push-Location $WorkingDirectory
    try {
        & $FilePath @ArgumentList
        if ($LASTEXITCODE -ne 0) { throw "$FilePath failed with exit code $LASTEXITCODE." }
    }
    finally { Pop-Location }
    Write-Host "[DONE] $display" -ForegroundColor Green
}

Write-Host "[START] Desktop artifact verification: $Variant $Version" -ForegroundColor Cyan
Update-VerificationProgress -Status 'Checking required artifacts'

function Invoke-ArtifactVerification {
    try {
        Assert-File $portable
        Assert-File $msi
        Assert-File $checksum
        Assert-File $metadataPath
        Assert-File $runtimeAudit
        Assert-File $python

        Update-VerificationProgress -Status 'Validating build metadata'
        $metadataJson = Get-Content -LiteralPath $metadataPath -Raw
        $metadata = $metadataJson | ConvertFrom-Json
        $metadataCreatedUtcMatch = [regex]::Match($metadataJson, '"created_utc"\s*:\s*"([^"]+)"')
        if (-not $metadataCreatedUtcMatch.Success) { throw 'Build metadata is missing a raw created_utc value.' }
        $metadataCreatedUtc = $metadataCreatedUtcMatch.Groups[1].Value
        $metadataProperties = @('format', 'application', 'version', 'variant', 'architecture', 'source_commit', 'created_utc', 'payload_sha256', 'artifacts', 'checksums')
        for ($index = 0; $index -lt $metadataProperties.Count; $index++) {
            $property = $metadataProperties[$index]
            Update-VerificationProgress -Status "Metadata field $($index + 1) of $($metadataProperties.Count): $property" -PercentComplete ([int](($index + 1) * 100 / $metadataProperties.Count))
            if ($null -eq $metadata.$property) { throw "Build metadata is missing '$property'." }
        }
        if ($metadata.format -ne 2 -or $metadata.application -ne 'XREPORT' -or $metadata.version -ne $Version -or
            $metadata.variant -ne $Variant -or $metadata.architecture -ne 'windows-x64' -or $metadata.source_commit -ne $SourceCommit) {
            throw "Build metadata does not match $Variant $Version $SourceCommit."
        }
        try {
            [DateTimeOffset]::ParseExact(
                $metadataCreatedUtc,
                "yyyy-MM-dd'T'HH:mm:ss'Z'",
                [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::AssumeUniversal
            ) | Out-Null
        }
        catch { throw 'Build metadata created_utc is invalid.' }
        $runtimeManifestJson = Get-Content -LiteralPath $runtimeAudit -Raw
        $runtimeManifest = $runtimeManifestJson | ConvertFrom-Json
        $runtimeCreatedUtcMatch = [regex]::Match($runtimeManifestJson, '"created_utc"\s*:\s*"([^"]+)"')
        if (-not $runtimeCreatedUtcMatch.Success) { throw 'Runtime audit is missing a raw created_utc value.' }
        $runtimeCreatedUtc = $runtimeCreatedUtcMatch.Groups[1].Value
        $runtimeProperties = @('format', 'application', 'version', 'variant', 'architecture', 'source_commit', 'created_utc', 'payload_sha256', 'backend_executable')
        for ($index = 0; $index -lt $runtimeProperties.Count; $index++) {
            $property = $runtimeProperties[$index]
            Update-VerificationProgress -Status "Runtime field $($index + 1) of $($runtimeProperties.Count): $property" -PercentComplete ([int](($index + 1) * 100 / $runtimeProperties.Count))
            if ($null -eq $runtimeManifest.$property) { throw "Runtime audit is missing '$property'." }
        }
        if ($runtimeManifest.format -ne 2 -or $runtimeManifest.application -ne 'XREPORT' -or
            $runtimeManifest.version -ne $Version -or $runtimeManifest.variant -ne $Variant -or
            $runtimeManifest.architecture -ne 'windows-x64' -or $runtimeManifest.source_commit -ne $SourceCommit -or
            $runtimeManifest.backend_executable -ne 'backend/XREPORT-backend.exe') {
            throw 'Runtime audit does not match the requested desktop build.'
        }
        if ($metadata.payload_sha256 -ne $runtimeManifest.payload_sha256 -or
            $metadataCreatedUtc -ne $runtimeCreatedUtc -or $metadata.source_commit -ne $runtimeManifest.source_commit -or
            $metadata.architecture -ne $runtimeManifest.architecture) {
            throw 'Build metadata does not match the runtime audit.'
        }

        Update-VerificationProgress -Status 'Validating artifact and checksum lists'
        $expectedArtifacts = @([IO.Path]::GetFileName($portable), [IO.Path]::GetFileName($msi)) | Sort-Object
        $actualArtifacts = @($metadata.artifacts | ForEach-Object { [string]$_ }) | Sort-Object
        if (($expectedArtifacts -join '|') -ne ($actualArtifacts -join '|')) { throw 'Build metadata artifact list is incomplete or stale.' }
        if ([IO.Path]::GetFileName($checksum) -ne [string]$metadata.checksums) { throw 'Build metadata checksum filename is stale.' }

        $checksumEntries = @{}
        $checksumLines = @(Get-Content -LiteralPath $checksum)
        for ($index = 0; $index -lt $checksumLines.Count; $index++) {
            $line = $checksumLines[$index]
            Update-VerificationProgress -Status "Reading checksum $($index + 1) of $($checksumLines.Count)" -PercentComplete ([int](($index + 1) * 100 / [Math]::Max(1, $checksumLines.Count)))
            if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') { throw "Invalid checksum line: $line" }
            $checksumEntries[$Matches[2]] = $Matches[1].ToLowerInvariant()
        }
        $artifactsToCheck = @($portable, $msi)
        for ($index = 0; $index -lt $artifactsToCheck.Count; $index++) {
            $artifact = $artifactsToCheck[$index]
            $name = [IO.Path]::GetFileName($artifact)
            Update-VerificationProgress -Status "Hashing artifact $($index + 1) of $($artifactsToCheck.Count): $name" -PercentComplete ([int](($index + 1) * 100 / $artifactsToCheck.Count))
            if (-not $checksumEntries.ContainsKey($name)) { throw "Checksum is missing $name." }
            $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifact).Hash.ToLowerInvariant()
            if ($actual -ne $checksumEntries[$name]) { throw "Checksum mismatch for $name." }
        }

        Update-VerificationProgress -Status 'Verifying portable runtime payload'
        $portableVerificationOutput = & $python @(
            $runtimeVerifier, '--portable', $portable, '--version', $Version, '--variant', $Variant,
            '--architecture', 'windows-x64', '--source-commit', $SourceCommit
        ) 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Portable runtime verification failed with exit code $LASTEXITCODE.`n$($portableVerificationOutput -join "`n")" }
        $portableVerification = ($portableVerificationOutput | Select-Object -Last 1) | ConvertFrom-Json
        if ($portableVerification.payload_sha256 -ne $runtimeManifest.payload_sha256) {
            throw 'Portable runtime payload does not match the variant runtime audit.'
        }

        Update-VerificationProgress -Status 'Inspecting MSI runtime resource'
        $installer = New-Object -ComObject WindowsInstaller.Installer
        $database = $installer.OpenDatabase((Resolve-Path -LiteralPath $msi).Path, 0)
        $view = $database.OpenView('SELECT FileName FROM File')
        $view.Execute()
        $fileNames = @()
        while ($record = $view.Fetch()) { $fileNames += [string]$record.StringData(1) }
        if (-not (@($fileNames | Where-Object { [string]$_ -match '(?i)(^|[\\/|])runtime\.zip$' }).Count)) {
            throw "MSI does not contain the generated runtime.zip resource."
        }

        $reportPath = Join-Path $repoRoot "assets\QA\desktop\verification-$Variant-$Version.json"
        New-Item -ItemType Directory -Path (Split-Path -Parent $reportPath) -Force | Out-Null
        [pscustomobject]@{
            format = 1
            application = 'XREPORT'
            version = $Version
            variant = $Variant
            architecture = 'windows-x64'
            source_commit = $SourceCommit
            portable = [IO.Path]::GetFileName($portable)
            msi = [IO.Path]::GetFileName($msi)
            runtime_resource = 'runtime.zip'
            verified_utc = [DateTime]::UtcNow.ToString('o')
        } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportPath -Encoding utf8
        Write-Host "Desktop artifacts verified: $Variant $Version" -ForegroundColor Green
    }
    finally {
        Write-Progress -Id $script:VerificationProgressId -Activity "XREPORT artifact verification: $Variant" -Completed
    }
}

Invoke-ArtifactVerification
