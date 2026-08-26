from __future__ import annotations

import json
from pathlib import Path

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from server.app import app

###############################################################################
SHARED_OPENAPI_PATH = (
    Path(__file__).resolve().parents[3] / "app" / "shared" / "openapi.json"
)

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
def test_shared_openapi_schema_matches_runtime() -> None:
    with SHARED_OPENAPI_PATH.open(encoding="utf-8") as schema_file:
        shared_schema = json.load(schema_file)

    assert shared_schema == app.openapi()

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
def test_health_endpoint_returns_backend_json() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "application": "XREPORT Backend",
        "version": "3.0.0",
        "runtime_mode": "sqlite",
        "runtime_variant": None,
        "runtime_port": None,
    }

###############################################################################
def test_inference_multipart_contract_excludes_removed_fields() -> None:
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

###############################################################################
def test_runtime_openapi_exposes_current_model_maintenance_paths() -> None:
    paths = app.openapi()["paths"]

    assert "/api/inference/models" in paths
    assert "/api/inference/models/check-update" in paths
    assert "/api/inference/models/maintenance" in paths
    assert "/api/inference/checkpoints" not in paths
