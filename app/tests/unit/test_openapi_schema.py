from __future__ import annotations

from fastapi.routing import APIRoute

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

###############################################################################
def test_stable_api_routes_declare_response_models() -> None:
    file_response_path = "/api/preparation/dataset/{dataset_name}/images/{index}/content"
    routes_without_models = {
        route.path
        for route in app.routes
        if isinstance(route, APIRoute)
        and route.path.startswith("/api/")
        and route.response_model is None
    }
    assert routes_without_models == {file_response_path}

###############################################################################
def test_inference_multipart_contract_excludes_legacy_fields() -> None:
    schema = app.openapi()
    request_schema = schema["paths"]["/api/inference/generate"]["post"][
        "requestBody"
    ]["content"]["multipart/form-data"]["schema"]
    component_name = request_schema["$ref"].rsplit("/", 1)[-1]
    properties = schema["components"]["schemas"][component_name]["properties"]

    assert set(properties) == {
        "model_ref",
        "generation_profile",
        "clinical_context",
        "images",
    }
    assert "checkpoint" not in properties
    assert "generation_mode" not in properties
