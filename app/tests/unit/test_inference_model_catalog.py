from __future__ import annotations

from server.configurations import InferenceSettings
from server.models.inference.catalog import InferenceModelCatalog


class ModelSerializerStub:
    def scan_checkpoints_folder(self) -> list[str]:
        return ["checkpoint_epoch_48"]


def _settings(*, hf_local_only: bool = True) -> InferenceSettings:
    return InferenceSettings(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_keep_alive="5m",
        hf_local_only=hf_local_only,
        hf_cache_dir=None,
        device="auto",
        max_loaded_models=1,
        model_timeout=600,
    )


def test_catalog_lists_only_curated_refs_and_discovered_xreport_checkpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.models.inference.catalog.ModelSerializer",
        ModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings()).list_models()

    assert [model.model_ref for model in response.models] == [
        "huggingface:google/medgemma-1.5-4b-it",
        "ollama:medgemma:4b",
        "ollama:medgemma:27b",
        "ollama:qwen3-vl:8b",
        "ollama:qwen3-vl:4b",
        "xreport:checkpoint_epoch_48",
    ]
    assert response.providers["ollama"].status == "runtime_unavailable"
    assert response.providers["huggingface"].status == "not_installed"
    assert response.providers["xreport"].status == "ready"


def test_catalog_disables_huggingface_when_local_only_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.models.inference.catalog.ModelSerializer",
        ModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings(hf_local_only=False)).list_models()

    medgemma = response.models[0]
    assert medgemma.status == "disabled"
    assert response.providers["huggingface"].status == "disabled"
