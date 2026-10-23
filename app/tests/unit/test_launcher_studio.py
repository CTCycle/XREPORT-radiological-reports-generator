from __future__ import annotations

from pathlib import Path
import re

LAUNCHER = (Path(__file__).parents[3] / "start_on_windows.ps1").read_text(
    encoding="utf-8"
)


def _function_body(name: str) -> str:
    match = re.search(rf"^function {re.escape(name)} \{{", LAUNCHER, re.MULTILINE)
    assert match, f"{name} not found"
    following = re.search(r"^function ", LAUNCHER[match.end():], re.MULTILINE)
    end = match.end() + following.start() if following else len(LAUNCHER)
    return LAUNCHER[match.start():end]


###############################################################################
def test_launch_actions_keep_the_launch_alias_and_add_studio() -> None:
    assert "'LaunchStudio'" in LAUNCHER and "'MigrateToStudio'" in LAUNCHER
    assert "'Launch' { Invoke-Launch -Application $App }" in LAUNCHER
    assert "'LaunchStudio' { Invoke-Launch -Application Studio }" in LAUNCHER
    launch = _function_body("Invoke-Launch")
    assert "'--app', 'studio'" in launch
    assert "STUDIO_FASTAPI_PORT" in launch
    assert "studio-angular" in launch


###############################################################################
def test_frontend_freshness_is_tracked_per_angular_project() -> None:
    state = _function_body("Write-FrontendBuildState")
    status = _function_body("Get-FrontendBuildStatus")
    fingerprint = _function_body("Get-FrontendBuildFingerprint")

    assert r"dist\$Project" in state and r"dist\$Project" in status
    assert "project=$Project" in fingerprint
    assert "build:studio" in _function_body("Get-FrontendBuildScript")


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
