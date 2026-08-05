from __future__ import annotations

import ast
from pathlib import Path

from fastapi.routing import APIRoute

from server.app import app


BACKEND_ROOT = Path(__file__).parents[2] / "server"
FILE_RESPONSE_PATH = "/api/preparation/dataset/{dataset_name}/images/{index}/content"


def _backend_files() -> list[Path]:
    return [
        path
        for path in BACKEND_ROOT.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    ]


def _imported_module(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name
    return node.module or ""


class _StructureVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports_inside_scopes: list[ast.Import | ast.ImportFrom] = []
        self.nested_functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.module_lock_assignments: list[ast.Assign | ast.AnnAssign] = []
        self._function_depth = 0
        self._class_depth = 0

    def visit_Import(self, node: ast.Import) -> None:
        if self._function_depth or self._class_depth:
            self.imports_inside_scopes.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._function_depth or self._class_depth:
            self.imports_inside_scopes.append(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._function_depth:
            self.nested_functions.append(node)
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._function_depth:
            self.nested_functions.append(node)
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_module_lock_call(node.value) and self._class_depth == 0:
            self.module_lock_assignments.append(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and self._is_module_lock_call(node.value):
            self.module_lock_assignments.append(node)
        self.generic_visit(node)

    @staticmethod
    def _is_module_lock_call(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"Lock", "RLock", "Event"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"threading", "asyncio"}
        )


def test_backend_structure_has_explicit_top_level_dependencies() -> None:
    violations: list[str] = []
    for path in _backend_files():
        source = path.read_text(encoding="utf-8")
        if len(source.splitlines()) > 1000:
            violations.append(f"{path}: exceeds 1000 lines")
        visitor = _StructureVisitor()
        visitor.visit(ast.parse(source, filename=str(path)))
        violations.extend(
            f"{path}: import inside scope at line {node.lineno}"
            for node in visitor.imports_inside_scopes
        )
        violations.extend(
            f"{path}: nested function at line {node.lineno}"
            for node in visitor.nested_functions
        )
        violations.extend(
            f"{path}: module-level mutable lock at line {node.lineno}"
            for node in visitor.module_lock_assignments
        )
        module_parts = path.relative_to(BACKEND_ROOT).parts
        layer = module_parts[0] if module_parts else ""
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            imported = _imported_module(node)
            forbidden = (
                (layer == "api" and imported.startswith(("server.repositories", "server.models")))
                or (layer == "services" and imported.startswith("server.api"))
                or (layer == "repositories" and imported.startswith(("server.api", "server.services")))
                or (layer == "models" and imported.startswith(("server.api", "server.services", "server.repositories")))
                or (path.name == "app.py" and imported.startswith("server.repositories"))
            )
            if forbidden:
                violations.append(f"{path}: forbidden import {imported} at line {node.lineno}")
    assert not violations, "\n".join(violations)


def test_backend_response_model_exception_is_only_file_response() -> None:
    routes_without_models = {
        route.path
        for route in app.routes
        if isinstance(route, APIRoute)
        and route.path.startswith("/api/")
        and route.response_model is None
    }
    assert routes_without_models == {FILE_RESPONSE_PATH}
