from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest
from fastapi import Request

from server.api.errors import handle_service_error
from server.services.errors import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    InternalServiceError,
    NotFoundError,
    PayloadTooLargeError,
    ServiceError,
    UnsupportedOperationError,
)


APP_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = APP_ROOT / "server"

###############################################################################
class BackendStructureVisitor(ast.NodeVisitor):

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.function_depth = 0
        self.local_imports: list[int] = []
        self.nested_functions: list[int] = []

    # -------------------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    # -------------------------------------------------------------------------
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    # -------------------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        if self.function_depth:
            self.local_imports.append(node.lineno)

    # -------------------------------------------------------------------------
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.function_depth:
            self.local_imports.append(node.lineno)

    # -------------------------------------------------------------------------
    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.function_depth:
            self.nested_functions.append(node.lineno)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

###############################################################################
def test_backend_module_boundaries_and_python_constraints() -> None:
    violations: list[str] = []
    source_paths = [SERVER_ROOT / "__init__.py", SERVER_ROOT / "app.py"]
    for package_name in (
        "api",
        "common",
        "configurations",
        "domain",
        "models",
        "repositories",
        "services",
    ):
        source_paths.extend((SERVER_ROOT / package_name).rglob("*.py"))

    for path in source_paths:
        relative_path = path.relative_to(APP_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        visitor = BackendStructureVisitor()
        visitor.visit(tree)

        if len(source.splitlines()) > 1000:
            violations.append(f"{relative_path}: exceeds 1000 lines")
        violations.extend(
            f"{relative_path}:{line}: import is not top-level"
            for line in visitor.local_imports
        )
        violations.extend(
            f"{relative_path}:{line}: nested function definition"
            for line in visitor.nested_functions
        )

        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        if relative_path.startswith("server/api/") and any(
            name.startswith("server.repositories") for name in imports
        ):
            violations.append(f"{relative_path}: API imports a repository")
        if relative_path.startswith("server/services/") and any(
            name == "fastapi" or name.startswith("server.api") for name in imports
        ):
            violations.append(f"{relative_path}: service imports the API layer")
        if relative_path.startswith("server/models/") and any(
            name.startswith(("server.api", "server.services", "server.repositories"))
            for name in imports
        ):
            violations.append(f"{relative_path}: model imports an outer layer")

    assert violations == []

###############################################################################
def test_obsolete_compatibility_modules_are_absent() -> None:
    obsolete_paths = [
        SERVER_ROOT / "models" / "training" / "worker.py",
        SERVER_ROOT / "services" / "processing.py",
        SERVER_ROOT / "models" / "inference" / "catalog.py",
    ]
    assert [path for path in obsolete_paths if path.exists()] == []

###############################################################################
@pytest.mark.parametrize(
    ("error_type", "expected_status"),
    [
        (BadRequestError, 400),
        (ForbiddenError, 403),
        (NotFoundError, 404),
        (ConflictError, 409),
        (PayloadTooLargeError, 413),
        (InternalServiceError, 500),
        (UnsupportedOperationError, 501),
    ],
)
def test_service_error_handler_preserves_http_contract(
    error_type: type[ServiceError], expected_status: int
) -> None:
    request = Request({"type": "http", "method": "GET", "path": "/"})
    response = asyncio.run(handle_service_error(request, error_type("stable detail")))

    assert response.status_code == expected_status
    assert json.loads(response.body) == {"detail": "stable detail"}
