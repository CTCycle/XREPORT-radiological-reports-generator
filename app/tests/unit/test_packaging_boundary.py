from __future__ import annotations

import ast
from pathlib import Path

from server.common.profile_modules import STUDIO_ONLY_MODULES

APP_ROOT = Path(__file__).resolve().parents[2]
ROOTS = ("server.app", "server.desktop_entry")


def _module_file(name: str) -> Path | None:
    base = APP_ROOT.joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _imports(path: Path, module: str) -> set[str]:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return {name for name in found if name.startswith("server")}


def _closure(roots: tuple[str, ...]) -> set[str]:
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        path = _module_file(module)
        if path is None:
            continue
        seen.add(module)
        parent = module.rpartition(".")[0]
        if parent.startswith("server"):
            pending.append(parent)
        pending.extend(_imports(path, module))
    return seen


###############################################################################
def test_studio_only_module_list_matches_real_modules() -> None:
    assert all(_module_file(name) is not None for name in STUDIO_ONLY_MODULES)


###############################################################################
def test_xreport_import_closure_never_reaches_studio_modules() -> None:
    reachable = _closure(ROOTS)
    leaked = sorted(set(STUDIO_ONLY_MODULES) & reachable)

    assert not leaked, f"XREPORT would package Studio-only modules: {leaked}"
    assert "server.services.inference_runtime" in reachable  # sanity: walk is real


###############################################################################
def test_desktop_build_excludes_studio_modules_and_assets() -> None:
    build = APP_ROOT / "desktop" / "build"
    spec = (build / "xreport_backend.spec").read_text(encoding="utf-8")
    bundle = (build / "create_runtime_bundle.py").read_text(encoding="utf-8")

    assert "STUDIO_ONLY_MODULES" in spec
    assert "excludes" in spec and "studio_excludes" in spec
    assert "studio-angular" in bundle
