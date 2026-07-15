from __future__ import annotations

import ast
from pathlib import Path


SERVER_ROOT = Path(__file__).parents[2] / "server"


def test_backend_modules_have_flat_imports_and_bounded_size() -> None:
    for path in SERVER_ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        source_lines = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert len(source_lines) <= 1000, f"{path} exceeds 1000 code lines"
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert not any(
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child is not node
                    for child in ast.walk(node)
                ), f"nested function in {path}:{node.lineno}"
            if isinstance(node, (ast.If, ast.Try, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                assert not any(
                    isinstance(child, (ast.Import, ast.ImportFrom))
                    for child in ast.iter_child_nodes(node)
                ), f"local or conditional import in {path}:{node.lineno}"
