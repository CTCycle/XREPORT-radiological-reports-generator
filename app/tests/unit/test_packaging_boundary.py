"""The two applications may only share code through `xreport_shared`."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("xreport_shared", "xreport_studio", "xreport_inference")
ALLOWED = {
    "xreport_shared": {"xreport_shared"},
    "xreport_studio": {"xreport_studio", "xreport_shared"},
    "xreport_inference": {"xreport_inference", "xreport_shared"},
}


def _python_files(package: str) -> list[Path]:
    return sorted(
        path for path in (APP_ROOT / package).rglob("*.py") if "__pycache__" not in path.parts
    )


def _application_references(path: Path) -> set[str]:
    """Top-level application packages named by imports or module-name strings."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [node.module or ""]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        found.update(
            name.split(".")[0] for name in names if name.split(".")[0] in PACKAGES
        )
    return found


###############################################################################
@pytest.mark.parametrize("package", PACKAGES)
def test_packages_only_depend_on_themselves_and_the_shared_package(package: str) -> None:
    violations = {
        path.relative_to(APP_ROOT).as_posix(): sorted(
            _application_references(path) - ALLOWED[package]
        )
        for path in _python_files(package)
        if _application_references(path) - ALLOWED[package]
    }

    assert not violations, f"{package} reaches into another application: {violations}"


###############################################################################
def test_legacy_server_package_is_gone() -> None:
    assert not (APP_ROOT / "server" / "__init__.py").exists()
    offenders = [
        path.relative_to(APP_ROOT).as_posix()
        for package in PACKAGES
        for path in _python_files(package)
        if any(
            line.strip().startswith(("from server.", "import server."))
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    ]

    assert not offenders, offenders


###############################################################################
def test_each_application_has_its_own_entry_point_and_migrations() -> None:
    for package, stream in (("xreport_studio", "studio"), ("xreport_inference", "inference")):
        assert (APP_ROOT / package / "app.py").is_file()
        assert (APP_ROOT / package / "entry.py").is_file()
        assert (APP_ROOT / package / "migrations" / stream / "env.py").is_file()
    assert (APP_ROOT / "xreport_inference" / "desktop_entry.py").is_file()
    assert not (APP_ROOT / "xreport_studio" / "desktop_entry.py").exists()


###############################################################################
def test_desktop_build_packages_only_the_inference_application() -> None:
    build = APP_ROOT / "desktop" / "build"
    spec = (build / "xreport_backend.spec").read_text(encoding="utf-8")
    bundle = (build / "create_runtime_bundle.py").read_text(encoding="utf-8")

    assert "xreport_inference" in spec and "xreport_shared" in spec
    assert 'collect_submodules("xreport_studio")' not in spec
    assert "studio-angular" in bundle
