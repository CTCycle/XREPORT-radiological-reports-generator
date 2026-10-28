from __future__ import annotations

import json
import pytest
import tomllib
from fastapi import Request
from tests.conftest import run_async_in_thread
from xreport_shared.api.errors import handle_service_error
from xreport_shared.services.errors import BadRequestError, ConflictError, ForbiddenError, InternalServiceError, NotFoundError, PayloadTooLargeError, ServiceError, UnsupportedOperationError
from pathlib import Path
from xreport_inference.app import app, create_app as create_inference_app
from xreport_studio.app import create_app as create_studio_app

CREATE_APP = {"xreport": create_inference_app, "studio": create_studio_app}
from xreport_shared.common.constants import FASTAPI_VERSION


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
    response = run_async_in_thread(
        handle_service_error(request, error_type("stable detail"))
    )

    assert response.status_code == expected_status
    assert json.loads(response.body) == {"detail": "stable detail"}







SHARED_DIR = Path(__file__).resolve().parents[3] / "app" / "shared"


###############################################################################
@pytest.mark.parametrize("profile", ["xreport", "studio"])
def test_shared_openapi_schema_matches_runtime(profile: str) -> None:
    schema_path = SHARED_DIR / f"openapi.{profile}.json"
    shared_schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert shared_schema == CREATE_APP[profile]().openapi()


###############################################################################
def test_schemas_defined_by_both_applications_are_identical() -> None:
    # The shared Angular code types jobs, settings and system status against one application's
    # generated schema, so the other application must expose exactly the same shapes.
    schemas = {
        profile: json.loads((SHARED_DIR / f"openapi.{profile}.json").read_text(encoding="utf-8"))["components"]["schemas"]
        for profile in CREATE_APP
    }
    common = set(schemas["xreport"]) & set(schemas["studio"])

    assert common
    assert [name for name in sorted(common) if schemas["xreport"][name] != schemas["studio"][name]] == []


###############################################################################
def test_inference_generate_contract_has_only_current_request_fields() -> None:
    schema = app.openapi()
    request_schema = schema["paths"]["/api/inference/generate"]["post"]["requestBody"][
        "content"
    ]["multipart/form-data"]["schema"]
    component_name = request_schema["$ref"].rsplit("/", 1)[-1]
    properties = schema["components"]["schemas"][component_name]["properties"]

    assert set(properties) == {
        "model_ref",
        "generation_profile",
        "clinical_context",
        "batch_mode",
        "images",
    }


###############################################################################
def test_workflow_contracts_require_client_selected_parameters() -> None:
    schemas = create_studio_app().openapi()["components"]["schemas"]

    assert schemas["LoadDatasetRequest"]["required"] == [
        "upload_id",
        "image_folder_path",
        "sample_size",
        "confirm_unmatched",
    ]
    assert schemas["StartTrainingRequest"]["required"] == [
        "dataset_name",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "architecture_preset",
        "use_img_augmentation",
        "save_checkpoints",
        "use_device_GPU",
        "device_ID",
        "use_mixed_precision",
        "dataloader_workers",
        "prefetch_factor",
        "pin_memory",
        "persistent_workers",
        "plot_training_metrics",
        "use_scheduler",
        "target_LR",
        "warmup_steps",
    ]
    assert schemas["ProcessDatasetRequest"]["required"] == [
        "dataset_name",
        "sample_size",
        "validation_size",
        "tokenizer",
        "max_report_size",
    ]
    assert schemas["ValidationRequest"]["required"] == [
        "dataset_name",
        "metrics",
        "sample_size",
    ]
    assert schemas["CheckpointEvaluationRequest"]["required"] == [
        "checkpoint",
        "metrics",
        "num_samples",
    ]








ROOT_DIR = Path(__file__).resolve().parents[3]


###############################################################################
def _json_version(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["version"])


###############################################################################
def test_release_version_has_one_canonical_source() -> None:
    with (ROOT_DIR / "app" / "server" / "pyproject.toml").open("rb") as pyproject_file:
        canonical = str(tomllib.load(pyproject_file)["project"]["version"])

    copies = [
        ROOT_DIR / "app" / "client" / "package.json",
        ROOT_DIR / "app" / "desktop" / "package.json",
        ROOT_DIR / "app" / "desktop" / "src-tauri" / "tauri.conf.json",
        ROOT_DIR / "app" / "desktop" / "src-tauri" / "tauri.cpu.conf.json",
        ROOT_DIR / "app" / "desktop" / "src-tauri" / "tauri.cuda.conf.json",
    ]

    assert FASTAPI_VERSION == canonical
    assert all(_json_version(path) == canonical for path in copies)

    openapi = json.loads(
        (ROOT_DIR / "app" / "shared" / "openapi.xreport.json").read_text(encoding="utf-8")
    )
    assert openapi["info"]["version"] == canonical
