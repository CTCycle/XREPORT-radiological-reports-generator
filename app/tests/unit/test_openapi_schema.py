from __future__ import annotations

from server.app import app

###############################################################################
def test_openapi_schema_generation_and_prefixes() -> None:
    schema = app.openapi()
    assert isinstance(schema, dict)

    paths = schema.get("paths", {})
    expected_prefixes = [
        "/api/upload",
        "/api/preparation",
        "/api/training",
        "/api/validation",
        "/api/inference",
    ]
    for prefix in expected_prefixes:
        assert any(path.startswith(prefix) for path in paths), f"Missing prefix: {prefix}"
