from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from server.domain.inference import InferenceManifest
from server.models.inference.providers.huggingface import HuggingFaceProvider
from server.services.inference_runtime import InferenceRuntimeCoordinator

###############################################################################
def _entry() -> dict[str, object]:
    return {
        "model_ref": "huggingface:aehrc/cxrmate-multi-tf",
        "repository_id": "aehrc/cxrmate-multi-tf",
        "provider": "huggingface",
        "enabled": True,
        "display_name": "CXRMate Multi TF",
        "description": "test",
        "category": "test",
        "license": "Apache-2.0",
        "revision": "6" * 40,
        "model_loader": "auto_model",
        "processor_loader": "auto",
        "adapter": "cxrmate_multi",
        "prompt_profile": "cxrmate-multi",
        "output_sections": ["findings", "impression"],
        "input_semantics": "single_study",
        "max_current_images": 16,
        "preferred_dtype": "auto",
        "quantization": ["none"],
        "validation_status": "pending",
        "required_files": ["config.json"],
        "weight_file_sets": [["model.safetensors"]],
    }

###############################################################################
def test_manifest_rejects_unknown_nested_fields() -> None:
    entry = _entry()
    entry["runtime_constraints"] = {"required_modules": [], "typo": True}

    with pytest.raises(ValidationError):
        InferenceManifest.model_validate({"schema_version": 3, "models": [entry] * 5})

###############################################################################
def test_manifest_rejects_empty_weight_alternative() -> None:
    entry = _entry()
    entry["weight_file_sets"] = [[]]

    with pytest.raises(ValidationError):
        InferenceManifest.model_validate({"schema_version": 3, "models": [entry] * 5})

###############################################################################
def test_runtime_rejects_provider_only_manifest() -> None:
    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        InferenceRuntimeCoordinator._require_complete_manifest(
            {"revision": "6" * 40}
        )

###############################################################################
def test_embedded_catalog_is_exactly_five_unique_sha_pinned_public_models() -> None:
    catalog_path = Path(__file__).parents[3] / "settings" / "inference_models.json"
    payload = json.loads(catalog_path.resolve().read_text(encoding="utf-8"))
    manifest = InferenceManifest.model_validate(payload)

    assert manifest.schema_version == 3
    assert len(manifest.models) == 5
    assert len({entry.model_ref for entry in manifest.models}) == 5
    assert all(len(entry.revision) == 40 for entry in manifest.models)
    assert {entry.adapter for entry in manifest.models} == {
        "cxrmate_multi",
        "cxrmate_ed",
        "chexone",
        "cxrmate2",
        "medgemma",
    }

###############################################################################
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adapter", "unsupported_adapter"),
        ("model_loader", "unsupported_model_loader"),
        ("processor_loader", "unsupported_processor_loader"),
    ],
)
def test_provider_rejects_unsupported_manifest_identifiers(field: str, value: str) -> None:
    manifest = _entry()
    manifest[field] = value

    with pytest.raises(RuntimeError, match="Unsupported"):
        HuggingFaceProvider.validate_manifest(
            "aehrc/cxrmate-multi-tf",
            manifest,
        )

###############################################################################
@pytest.mark.parametrize("field", ["adapter", "model_loader", "processor_loader", "max_current_images", "preferred_dtype"])
def test_provider_requires_complete_manifest_fields(field: str) -> None:
    manifest = _entry()
    del manifest[field]

    with pytest.raises(RuntimeError, match="missing required field"):
        HuggingFaceProvider.validate_manifest(
            "aehrc/cxrmate-multi-tf",
            manifest,
        )
