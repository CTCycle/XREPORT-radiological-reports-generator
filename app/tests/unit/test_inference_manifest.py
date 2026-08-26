from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.domain.inference import InferenceManifest
from server.services.inference_runtime import InferenceRuntimeCoordinator


###############################################################################
def test_embedded_catalog_contains_exactly_five_unique_sha_pinned_public_models() -> None:
    catalog_path = Path(__file__).parents[3] / "settings" / "inference_models.json"
    manifest = InferenceManifest.model_validate(
        json.loads(catalog_path.resolve().read_text(encoding="utf-8"))
    )

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
def test_runtime_rejects_incomplete_model_manifest() -> None:
    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        InferenceRuntimeCoordinator._require_complete_manifest({"revision": "6" * 40})
