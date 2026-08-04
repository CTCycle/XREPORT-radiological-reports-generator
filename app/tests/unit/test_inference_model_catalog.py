from __future__ import annotations

import pytest

from server.configurations import InferenceSettings
from server.services.inference_catalog import InferenceModelCatalog
from server.services.model_installation import ModelInstallationManager
from server.common.path import ROOT_DIR


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


@pytest.fixture(autouse=True)
def isolate_project_installations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep catalog unit tests independent from a real local model install."""
    monkeypatch.setattr(
        ModelInstallationManager,
        "inspect",
        lambda _self, _manifest: {
            "metadata": {},
            "state": "not_installed",
            "integrity": "unknown",
            "active_revision": None,
            "active_path": None,
            "candidate": None,
            "candidate_path": None,
            "candidate_revision": None,
        },
    )

###############################################################################
def test_catalog_lists_only_supported_external_model_and_xreport_refs(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        ModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings()).list_models()

    assert [model.model_ref for model in response.models] == [
        "huggingface:nathansutton/generate-cxr",
        "xreport:checkpoint_epoch_48",
    ]
    assert "ollama" not in response.providers
    assert response.providers["huggingface"].status == "not_installed"
    assert response.providers["xreport"].status == "ready"
    assert response.models[0].status == "not_installed"
    assert response.models[0].installation_state == "not_installed"
    assert response.models[0].available_actions == []
    xreport = response.models[-1]
    assert xreport.output_sections == ["raw_report"]

###############################################################################
def test_catalog_disables_huggingface_when_local_only_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        ModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings(hf_local_only=False)).list_models()

    generate_cxr = next(model for model in response.models if model.model_ref.endswith("nathansutton/generate-cxr"))
    assert generate_cxr.status == "disabled"
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
    response = InferenceModelCatalog(_settings()).list_models()

    generate_cxr = next(model for model in response.models if model.model_ref.endswith("nathansutton/generate-cxr"))
    assert generate_cxr.model_revision == "6609ed3b711769816141f0f6fdaa88310e1ea0cb"
    assert generate_cxr.model_loader == "blip_conditional_generation"
    assert generate_cxr.processor_loader == "blip"
    assert generate_cxr.adapter == "generate_cxr_blip"
    assert generate_cxr.output_sections == ["raw_report"]
    assert generate_cxr.max_current_images == 1
    assert generate_cxr.license == "Apache-2.0"
    assert generate_cxr.status == "not_installed"

###############################################################################
def test_catalog_does_not_mark_unvalidated_cached_candidate_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        EmptyModelSerializerStub,
    )
    monkeypatch.setattr(
        "server.services.inference_catalog.HuggingFaceProvider.is_cached",
        lambda self, repository_id, revision, **kwargs: True,
    )

    response = InferenceModelCatalog(_settings()).list_models()

    assert response.models[0].status != "ready"


def test_catalog_distinguishes_verified_active_and_candidate_installations(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        EmptyModelSerializerStub,
    )
    active_path = ROOT_DIR / "app" / "resources" / "models" / "huggingface" / "installed" / "active"
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelInstallationManager.inspect",
        lambda self, manifest: {
            "metadata": {},
            "state": "active",
            "integrity": "verified",
            "active_revision": manifest["revision"],
            "active_path": active_path,
            "candidate": None,
            "candidate_path": None,
            "candidate_revision": None,
        },
    )
    response = InferenceModelCatalog(_settings()).list_models()
    model = response.models[0]
    assert model.status == "ready"
    assert model.installation_state == "active"
    assert model.integrity_status == "verified"
    assert model.local_path == "app/resources/models/huggingface/installed/active"
