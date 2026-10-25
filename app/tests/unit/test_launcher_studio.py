from __future__ import annotations

from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).parents[3]
LAUNCHER = (ROOT / "start_on_windows.ps1").read_text(
    encoding="utf-8"
)
TEST_RUNNER = (ROOT / "app/tests/run_tests.bat").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "app/server/pyproject.toml").read_text(encoding="utf-8"))
LOCK = tomllib.loads((ROOT / "app/server/uv.lock").read_text(encoding="utf-8"))


def _function_body(name: str) -> str:
    match = re.search(rf"^function {re.escape(name)} \{{", LAUNCHER, re.MULTILINE)
    assert match, f"{name} not found"
    following = re.search(r"^function ", LAUNCHER[match.end():], re.MULTILINE)
    end = match.end() + following.start() if following else len(LAUNCHER)
    return LAUNCHER[match.start():end]


def _distribution_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()


###############################################################################
def test_launch_actions_keep_the_launch_alias_and_add_studio() -> None:
    assert "'LaunchStudio'" in LAUNCHER
    assert "MigrateToStudio" not in LAUNCHER
    assert "MigrateStudio" not in LAUNCHER
    assert "'Launch' { Invoke-Launch -Application $App }" in LAUNCHER
    assert "'LaunchStudio' { Invoke-Launch -Application Studio }" in LAUNCHER
    launch = _function_body("Invoke-Launch")
    assert "'--app', 'studio'" in launch
    assert "STUDIO_FASTAPI_PORT" in launch
    assert "studio-angular" in launch
    assert "$env:XREPORT_APP" in launch


###############################################################################
def test_frontend_freshness_is_tracked_per_angular_project() -> None:
    state = _function_body("Write-FrontendBuildState")
    status = _function_body("Get-FrontendBuildStatus")
    fingerprint = _function_body("Get-FrontendBuildFingerprint")

    assert r"dist\$Project" in state and r"dist\$Project" in status
    assert "project=$Project" in fingerprint
    assert "build:studio" in _function_body("Get-FrontendBuildScript")


###############################################################################
def test_frontend_freshness_validates_toolchain_and_built_outputs() -> None:
    fingerprint = _function_body("Get-FrontendBuildFingerprint")
    dependency_fingerprint = _function_body("Get-FrontendDependencyFingerprint")
    output_manifest = _function_body("Get-FrontendOutputManifest")
    output_check = _function_body("Test-FrontendOutputManifestCurrent")
    write_state = _function_body("Write-FrontendBuildState")
    status = _function_body("Get-FrontendBuildStatus")
    ensure = _function_body("Ensure-FrontendBuild")
    rebuild = _function_body("Invoke-RebuildFrontend")

    assert '"npm=$NpmVersion"' in fingerprint
    assert '"npm=$NpmVersion"' in dependency_fingerprint
    assert "Get-FileHash" in output_manifest
    assert "ExpectedFiles[$index].sha256" in output_check
    assert "npm_version = $NpmVersion" in write_state
    assert "output_files = @($OutputFiles)" in write_state
    assert "state.npm_version" in status
    assert "Test-FrontendOutputManifestCurrent" in status
    assert "'NpmVersionChanged'" in ensure
    assert "Invoke-FrontendBuild -Project $project" in rebuild


###############################################################################
def test_environment_template_contains_no_obsolete_or_unused_settings() -> None:
    template = (ROOT / "settings/.env.example").read_text(encoding="utf-8")
    keys = {
        line.split("=", maxsplit=1)[0].strip()
        for line in template.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", ";")) and "=" in line
    }

    assert "RELOAD" not in keys
    assert not keys.intersection(
        {
            "ALWAYS_REBUILD",
            "HF_LOCAL_ONLY",
            "INFERENCE_MAX_LOADED_MODELS",
            "INFERENCE_MODEL_TIMEOUT",
        }
    )
    assert {
        "FASTAPI_HOST",
        "FASTAPI_PORT",
        "STUDIO_FASTAPI_PORT",
        "CORS_ALLOWED_ORIGINS",
        "INFERENCE_HF_LOCAL_ONLY",
        "INFERENCE_DEVICE",
        "HF_TOKEN",
        "XREPORT_RESOURCES_DIR",
    }.issubset(keys)


###############################################################################
def test_process_ownership_is_scoped_per_application() -> None:
    identify = _function_body("Get-XReportApplicationProcessIds")
    kill = _function_body("Stop-XReportProcesses")

    assert r"--app\s+studio" in identify
    assert "-not $isStudioBackend" in identify
    assert "STUDIO_FASTAPI_PORT" in kill
    assert "Get-XReportApplicationProcessIds -ProcessTable $processTable -Scope $Scope" in kill


###############################################################################
def test_cleanup_targets_never_cross_application_boundaries() -> None:
    targets = _function_body("Get-XReportUserDataTargets")
    studio_branch, xreport_branch = targets.split("if ($CheckpointsOnly) {\n        return @((Join-Path $resourceRoot", 1)

    assert "'studio'" in studio_branch
    assert "'models'" not in studio_branch
    assert "'studio'" not in xreport_branch
    assert "Join-Path $resourceRoot 'models'" in xreport_branch


###############################################################################
def test_desktop_build_stays_inference_only() -> None:
    for name in (
        "Invoke-DesktopFrontendBuild",
        "Invoke-DesktopBackendFreeze",
        "Invoke-DesktopVariantBuild",
    ):
        assert "studio" not in _function_body(name).lower()


###############################################################################
def test_studio_installation_uses_an_isolated_profiled_environment() -> None:
    assert "'InstallStudio'" in LAUNCHER
    assert "[ValidateSet('Standard', 'Cuda')]" in LAUNCHER
    assert "$StudioVenvDir = Join-Path $ServerDir '.venv-studio'" in LAUNCHER
    assert "$StudioVenvPython = Join-Path $StudioVenvDir 'Scripts\\python.exe'" in LAUNCHER

    launch = _function_body("Invoke-Launch")
    assert "Test-StudioBackendDependenciesReady -Profile $selectedStudioProfile" in launch
    assert "$backendPython = $StudioVenvPython" in launch
    assert "$backendPython = $VenvPython" in launch
    assert "Start-Process -FilePath $backendPython" in launch
    assert "Install-StudioBackendDependencies -Profile $selectedStudioProfile" in launch

    install = _function_body("Install-StudioBackendDependencies")
    assert "Get-StudioDependencyFingerprint -Profile $Profile" in install
    assert "$env:UV_PROJECT_ENVIRONMENT = $StudioVenvDir" in install
    assert "'sync', '--locked'" in install
    assert "'--extra', 'studio', '--extra', $profileExtra" in install
    assert "Move-Item -LiteralPath $temporaryPath -Destination $StudioInstallStatePath -Force" in install
    install_action = _function_body("Invoke-InstallStudio")
    assert "Stop-XReportProcesses -Scope Studio" in install_action
    assert "Ensure-FrontendBuild -Status $buildStatus -Project 'studio-angular'" in install_action
    assert "'InstallStudio'; Label = 'Install / update Studio'" in LAUNCHER


###############################################################################
def test_studio_profile_defaults_and_failure_retry_are_explicit() -> None:
    prompt = _function_body("Read-StudioProfile")
    resolve = _function_body("Resolve-StudioProfile")
    install = _function_body("Install-StudioBackendDependencies")

    assert "DefaultProfile = 'Standard'" in prompt
    assert "Select XREPORT Studio profile" in prompt
    assert "if (-not $script:LauncherInteractive) { return $DefaultProfile }" in prompt
    assert "if ($StudioProfileExplicit) { return $StudioProfile }" in resolve
    assert "Test-Path -LiteralPath $StudioVenvPython -PathType Leaf" in resolve
    assert "if ($installedProfile) { return $installedProfile }" in resolve
    assert "if ($script:LauncherInteractive) { return Read-StudioProfile -DefaultProfile 'Standard' }" in resolve
    assert "return 'Standard'" in resolve
    assert "[string]$state.profile -ne $Profile" in install
    assert "[string]$state.dependency_fingerprint -ne $fingerprint" in install
    assert "Remove-LauncherPath -Path $StudioVenvDir" in install
    assert "catch {" in install and "Remove-Item -LiteralPath $StudioInstallStatePath" in install
    assert "Test-StudioPythonEnvironment -Profile $Profile" in _function_body("Test-StudioBackendDependenciesReady")
    assert "Test-Path -LiteralPath $PythonExe" in _function_body("Test-StudioBackendDependenciesReady")
    assert "$StudioVenvDir" in _function_body("Uninstall-Application")


###############################################################################
def test_live_suite_uses_the_separate_studio_backend_environment() -> None:
    assert 'set "STUDIO_VENV_PYTHON=%SERVER_DIR%\\.venv-studio\\Scripts\\python.exe"' in TEST_RUNNER
    assert '"%STUDIO_VENV_PYTHON%" -m uvicorn' in TEST_RUNNER
    assert "start_on_windows.ps1 -Action InstallStudio" in TEST_RUNNER


###############################################################################
def test_xreport_setup_keeps_cuda_and_omits_studio_for_standard_installs() -> None:
    install = _function_body("Install-BackendDependencies")
    assert "'--extra', 'cuda'" in install
    standard_args = install.split("if ($InstallationType -eq 'Development')", 1)[0]
    assert "'--extra', 'studio'" not in standard_args
    assert "if ($InstallationType -eq 'Development')" in install
    assert "elseif ($InstallationType -eq 'Desktop')" in install
    assert "'--all-extras'" not in install
    assert "'--extra', 'studio'" in install


###############################################################################
def test_dependency_profiles_keep_studio_only_packages_optional() -> None:
    project = PYPROJECT["project"]
    dependencies = {_distribution_name(dependency) for dependency in project["dependencies"]}
    extras = project["optional-dependencies"]
    profiles = {name: {_distribution_name(dependency) for dependency in values} for name, values in extras.items()}

    assert not {"torch", "torchvision", "accelerate", "timm", "opencv-python", "matplotlib"} & dependencies
    assert {"accelerate", "torch", "torchvision", "timm"} <= profiles["standard"]
    assert {"accelerate", "torch", "torchvision", "timm"} <= profiles["cuda"]
    assert {"opencv-python", "matplotlib"} <= profiles["studio"]

    conflicts = PYPROJECT["tool"]["uv"]["conflicts"]
    assert [{"extra": "standard"}, {"extra": "cuda"}] in conflicts
    sources = PYPROJECT["tool"]["uv"]["sources"]
    for package in ("torch", "torchvision"):
        assert {(source["extra"], source["index"]) for source in sources[package]} == {
            ("standard", "pytorch-cpu"),
            ("cuda", "pytorch-cu130"),
        }


###############################################################################
def test_lock_contains_both_profile_torch_builds() -> None:
    packages = LOCK["package"]
    torch_versions = {
        package["version"]
        for package in packages
        if package["name"] == "torch"
    }
    torchvision_versions = {
        package["version"]
        for package in packages
        if package["name"] == "torchvision"
    }

    assert torch_versions == {"2.10.0+cpu", "2.10.0+cu130"}
    assert torchvision_versions == {"0.25.0+cpu", "0.25.0+cu130"}
