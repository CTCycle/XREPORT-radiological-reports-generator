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

###############################################################################
def test_workflow_contracts_require_client_selected_parameters() -> None:
    schemas = app.openapi()["components"]["schemas"]

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
        "num_encoders",
        "num_decoders",
        "embedding_dims",
        "attention_heads",
        "train_temp",
        "freeze_img_encoder",
        "use_img_augmentation",
        "shuffle_with_buffer",
        "shuffle_size",
        "save_checkpoints",
        "use_device_GPU",
        "device_ID",
        "jit_compile",
        "jit_backend",
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
