from __future__ import annotations

from server.configurations import InferenceSettings
from server.services.inference_catalog import InferenceModelCatalog


REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"

###############################################################################
class ModelSerializerStub:

    # -------------------------------------------------------------------------
    def scan_checkpoints_folder(self) -> list[str]:
        return ["checkpoint_epoch_48"]

###############################################################################
class EmptyModelSerializerStub:

    # -------------------------------------------------------------------------
    def scan_checkpoints_folder(self) -> list[str]:
        return []

###############################################################################
def _settings(*, hf_local_only: bool = True) -> InferenceSettings:
    return InferenceSettings(
        hf_local_only=hf_local_only,
        hf_cache_dir=None,
        device="auto",
        max_loaded_models=1,
        model_timeout=600,
    )

###############################################################################
def test_catalog_lists_only_embedded_and_xreport_refs(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        ModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings()).list_models()

    assert [model.model_ref for model in response.models] == [
        "huggingface:google/medgemma-1.5-4b-it",
        "xreport:checkpoint_epoch_48",
    ]
    assert "ollama" not in response.providers
    assert response.providers["huggingface"].status == "not_installed"
    assert response.providers["xreport"].status == "ready"

###############################################################################
def test_catalog_disables_huggingface_when_local_only_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        ModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings(hf_local_only=False)).list_models()

    medgemma = response.models[0]
    assert medgemma.status == "disabled"
    assert response.providers["huggingface"].status == "disabled"

###############################################################################
def test_catalog_marks_xreport_unavailable_without_complete_checkpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        EmptyModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings()).list_models()

    assert not any(model.provider == "xreport" for model in response.models)
    assert response.providers["xreport"].status == "not_installed"

###############################################################################
def test_catalog_uses_manifest_revision_and_exposes_runtime_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        ModelSerializerStub,
    )
    monkeypatch.setattr(
        "server.services.inference_catalog.HuggingFaceProvider.is_cached",
        lambda self, repository_id, revision: repository_id == "google/medgemma-1.5-4b-it" and revision == REVISION,
    )

    response = InferenceModelCatalog(_settings()).list_models()

    medgemma = response.models[0]
    assert medgemma.model_revision == REVISION
    assert medgemma.model_loader == "image_text_to_text"
    assert medgemma.processor_loader == "auto"
    assert medgemma.adapter == "medgemma"
    assert medgemma.output_sections == ["findings", "impression"]
    assert medgemma.max_current_images == 1
    assert medgemma.license == "Health AI Developer Foundation terms of use"
    assert medgemma.status == "incompatible"
    assert "not operational" in (medgemma.status_message or "")

###############################################################################
def test_catalog_does_not_mark_unvalidated_cached_candidate_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        EmptyModelSerializerStub,
    )
    monkeypatch.setattr(
        "server.services.inference_catalog.HuggingFaceProvider.is_cached",
        lambda self, repository_id, revision: True,
    )

    response = InferenceModelCatalog(_settings()).list_models()

    assert response.models[0].status != "ready"
