from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.domain.inference import InferenceManifest
from server.services.inference_runtime import InferenceRuntimeCoordinator


def _entry() -> dict[str, object]:
    return {
        "model_ref": "huggingface:nathansutton/generate-cxr",
        "repository_id": "nathansutton/generate-cxr",
        "provider": "huggingface",
        "enabled": True,
        "display_name": "generate-cxr",
        "description": "test",
        "category": "test",
        "license": "Apache-2.0",
        "revision": "6" * 40,
        "model_loader": "blip_conditional_generation",
        "processor_loader": "blip",
        "adapter": "generate_cxr_blip",
        "prompt_profile": "indication_prefix",
        "output_sections": ["raw_report"],
        "input_semantics": "single_image",
        "max_current_images": 1,
        "quantization": ["none"],
        "validation_status": "pending",
        "required_files": ["config.json"],
        "weight_file_sets": [["model.safetensors"]],
    }


def test_manifest_rejects_unknown_nested_fields() -> None:
    entry = _entry()
    entry["runtime_constraints"] = {"required_modules": [], "typo": True}

    with pytest.raises(ValidationError):
        InferenceManifest.model_validate({"schema_version": 2, "models": [entry]})


def test_manifest_rejects_empty_weight_alternative() -> None:
    entry = _entry()
    entry["weight_file_sets"] = [[]]

    with pytest.raises(ValidationError):
        InferenceManifest.model_validate({"schema_version": 2, "models": [entry]})


def test_runtime_rejects_provider_only_manifest() -> None:
    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        InferenceRuntimeCoordinator._require_complete_manifest(
            {"revision": "6" * 40}
        )
