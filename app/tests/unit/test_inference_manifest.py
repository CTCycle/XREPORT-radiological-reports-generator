from __future__ import annotations

import pytest

from server.configurations.inference_models import embedded_inference_models
from server.services.inference_runtime import InferenceRuntimeCoordinator
from server.services.inference_catalog import validation_contract_hash

###############################################################################
def test_embedded_catalog_contains_exactly_five_unique_sha_pinned_public_models() -> (
    None
):
    models = embedded_inference_models()
    assert len(models) == 5
    assert len({entry.model_ref for entry in models}) == 5
    assert all(len(entry.revision) == 40 for entry in models)
    assert {entry.adapter for entry in models} == {
        "cxrmate_multi",
        "cxrmate_ed",
        "chexone",
        "cxrmate2",
        "medgemma",
    }
    assert {entry.model_ref: entry.revision for entry in models} == {
        "huggingface:aehrc/cxrmate-multi-tf": "330721b9aa5bba201a3eb88eba4dd9a6607f3e7a",
        "huggingface:aehrc/cxrmate-ed": "68251c7605067ddbea330413aade032713fd2192",
        "huggingface:StanfordAIMI/CheXOne": "0c350e6852ea08f9d9baf3b7595c1a10d4849927",
        "huggingface:aehrc/cxrmate-2": "aa8e2d16470e20671acf049687b4707c9bf2f2b5",
        "huggingface:google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    }
    assert {
        entry.model_ref: validation_contract_hash(entry)
        for entry in models
    } == {
        "huggingface:aehrc/cxrmate-multi-tf": "770983f4b6d02fe550d739106838a4834719dad0b0d54258c9738790af0d0ddd",
        "huggingface:aehrc/cxrmate-ed": "d7e426e7df0ee447fe84f0dfcd57cab4e16c9b5e4d72cdef94142784ad0584fe",
        "huggingface:StanfordAIMI/CheXOne": "0b55bfd6edd786b14ba7a7af198caeb5c9f5759c797cf72cd8c6c75c19497dc9",
        "huggingface:aehrc/cxrmate-2": "ef9bf48a66e2e6a6df964da4fbcef1d9c3be0d783ee4a95376f900650e4c59ca",
        "huggingface:google/medgemma-1.5-4b-it": "db8398f1c7a5dfd7d1c5b41a67cb67109f9e77865dc7ad3836cc86bd796b761d",
    }

###############################################################################
def test_runtime_rejects_incomplete_model_manifest() -> None:
    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        InferenceRuntimeCoordinator._require_complete_manifest({"revision": "6" * 40})
