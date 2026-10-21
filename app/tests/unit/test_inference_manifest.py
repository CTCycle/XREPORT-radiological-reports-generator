from __future__ import annotations

import pytest

from server.configurations.inference_models import embedded_inference_models
from server.services.inference_runtime import InferenceRuntimeCoordinator
from server.services.inference_catalog import validation_contract_hash

###############################################################################
def test_embedded_catalog_contains_unique_sha_pinned_public_models() -> None:
    models = embedded_inference_models()
    assert models
    assert len({entry.model_ref for entry in models}) == len(models)
    assert all(len(entry.revision) == 40 for entry in models)
    assert {entry.adapter for entry in models} == {
        "cxrmate_multi",
        "chexone",
        "cxrmate2",
        "medgemma",
    }
    assert {entry.model_ref: entry.revision for entry in models} == {
        "huggingface:aehrc/cxrmate-multi-tf": "330721b9aa5bba201a3eb88eba4dd9a6607f3e7a",
        "huggingface:StanfordAIMI/CheXOne": "0c350e6852ea08f9d9baf3b7595c1a10d4849927",
        "huggingface:aehrc/cxrmate-2": "aa8e2d16470e20671acf049687b4707c9bf2f2b5",
        "huggingface:google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    }
    assert {
        entry.model_ref: validation_contract_hash(entry)
        for entry in models
    } == {
        "huggingface:aehrc/cxrmate-multi-tf": "4feea8f90fba6c26881d35e6178fcc878eb56a8ed53df717d74f856d6df92ee7",
        "huggingface:StanfordAIMI/CheXOne": "06314a6921f88ddecbaa25cd9eef44e0cbab78701b06319de11feffbd930dc4e",
        "huggingface:aehrc/cxrmate-2": "cc03947d5f74291f602c9c05e341d58d43fab9883f18637278fe35ee24f0a1c1",
        "huggingface:google/medgemma-1.5-4b-it": "a22ad3bcad11b2438bee9fc4bc53d1cb127d81b42931b0e1192e6170418c0163",
    }

###############################################################################
def test_embedded_chexone_manifest_declares_findings_only() -> None:
    chexone = next(
        entry
        for entry in embedded_inference_models()
        if entry.model_ref == "huggingface:StanfordAIMI/CheXOne"
    )

    assert chexone.output_sections == ["findings"]
    assert chexone.capabilities.findings is True
    assert chexone.capabilities.impression is False
    assert chexone.capabilities.grounding is True

###############################################################################
def test_runtime_rejects_incomplete_model_manifest() -> None:
    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        InferenceRuntimeCoordinator._require_complete_manifest({"revision": "6" * 40})
