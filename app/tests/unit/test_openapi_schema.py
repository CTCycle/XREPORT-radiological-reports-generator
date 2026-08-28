from __future__ import annotations

import json
from pathlib import Path

from server.app import app


SHARED_OPENAPI_PATH = Path(__file__).resolve().parents[3] / "app" / "shared" / "openapi.json"

###############################################################################
def test_shared_openapi_schema_matches_runtime() -> None:
    with SHARED_OPENAPI_PATH.open(encoding="utf-8") as schema_file:
        shared_schema = json.load(schema_file)

    assert shared_schema == app.openapi()

###############################################################################
def test_inference_generate_contract_has_only_current_request_fields() -> None:
    schema = app.openapi()
    request_schema = schema["paths"]["/api/inference/generate"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    component_name = request_schema["$ref"].rsplit("/", 1)[-1]
    properties = schema["components"]["schemas"][component_name]["properties"]

    assert set(properties) == {
        "model_ref",
        "generation_profile",
        "clinical_context",
        "images",
    }
