"""The Angular applications may only share code through `projects/shared`."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

CLIENT_ROOT = Path(__file__).resolve().parents[2] / "client"
PROJECTS = ("shared", "studio", "inference")
IMPORT = re.compile(r"""(?:from\s+|import\s*\(\s*|import\s+)['"]([^'"]+)['"]""")
STYLE = re.compile(r"""(?:styleUrls?|templateUrl)\s*:\s*(\[[^\]]*\]|['"][^'"]+['"])""")


def _sources(project: str) -> list[Path]:
    return sorted((CLIENT_ROOT / "projects" / project / "src").rglob("*.ts"))


def _references(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    styles = [item for match in STYLE.finditer(text) for item in re.findall(r"""['"]([^'"]+)['"]""", match.group(1))]
    return [*IMPORT.findall(text), *styles]


###############################################################################
@pytest.mark.parametrize("project", PROJECTS)
def test_projects_only_reach_their_own_sources_or_the_shared_alias(project: str) -> None:
    root = (CLIENT_ROOT / "projects" / project).resolve()
    violations: list[str] = []
    for path in _sources(project):
        for reference in _references(path):
            if reference.startswith("@xreport/"):
                if not reference.startswith("@xreport/shared/"):
                    violations.append(f"{path.name}: {reference}")
                continue
            if not reference.startswith("."):
                continue
            target = (path.parent / reference).resolve()
            # Component styles cannot use the TypeScript alias, so shared CSS is referenced by path.
            shared_style = reference.endswith(".css") and (CLIENT_ROOT / "projects" / "shared").resolve() in target.parents
            if root not in (target, *target.parents) and not shared_style:
                violations.append(f"{path.relative_to(CLIENT_ROOT).as_posix()}: {reference}")

    assert not violations, f"{project} reaches outside its project: {violations}"


###############################################################################
def test_shared_sources_do_not_use_application_specific_aliases() -> None:
    assert CLIENT_ROOT.joinpath("projects", "shared").is_dir()
    assert not (CLIENT_ROOT / "src").exists()


###############################################################################
def test_each_application_has_its_own_entry_point_routes_and_proxy() -> None:
    inference = CLIENT_ROOT / "projects" / "inference" / "src"
    studio = CLIENT_ROOT / "projects" / "studio" / "src"

    for root in (inference, studio):
        assert (root / "main.ts").is_file() and (root / "index.html").is_file()
    assert (inference / "app" / "app.routes.ts").is_file()
    assert (studio / "studio.routes.ts").is_file()
    assert (inference / "proxy.conf.cjs").is_file()
    assert (studio / "proxy.studio.conf.cjs").is_file()


###############################################################################
def test_angular_workspace_and_scripts_only_reference_existing_files_of_their_own_application() -> None:
    workspace = json.loads((CLIENT_ROOT / "angular.json").read_text(encoding="utf-8"))["projects"]
    other = {"client-angular": "studio", "studio-angular": "inference"}
    for name, forbidden in other.items():
        text = json.dumps(workspace[name])
        assert f"projects/{forbidden}/" not in text, f"{name} references the {forbidden} project"
        assert f"{forbidden}-angular" not in text, f"{name} builds through the {forbidden} application"

    scripts = json.loads((CLIENT_ROOT / "package.json").read_text(encoding="utf-8"))["scripts"]
    for command in scripts.values():
        for relative in re.findall(r"--proxy-config\s+(\S+)", command):
            assert (CLIENT_ROOT / relative).is_file(), f"script references a missing file: {relative}"


###############################################################################
def test_client_sources_have_no_circular_imports() -> None:
    graph: dict[Path, set[Path]] = {}
    for project in PROJECTS:
        for path in _sources(project):
            edges: set[Path] = set()
            for reference in _references(path):
                if reference.startswith("@xreport/shared/"):
                    base = CLIENT_ROOT / "projects" / "shared" / "src" / reference.removeprefix("@xreport/shared/")
                elif reference.startswith("."):
                    base = (path.parent / reference).resolve()
                else:
                    continue
                for candidate in (base.with_suffix(".ts"), base / "index.ts"):
                    if candidate.is_file():
                        edges.add(candidate.resolve())
            graph[path.resolve()] = edges

    state: dict[Path, int] = {}
    cycles: list[str] = []

    def visit(node: Path, trail: list[Path]) -> None:
        state[node] = 1
        for target in graph.get(node, ()):
            if state.get(target) == 1:
                cycles.append(" -> ".join(item.name for item in [*trail, node, target]))
            elif target not in state:
                visit(target, [*trail, node])
        state[node] = 2

    for node in graph:
        if node not in state:
            visit(node, [])

    assert not cycles
