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
        device="auto",
        max_loaded_models=1,
        model_timeout=600,
    )

###############################################################################
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
def test_catalog_lists_exactly_five_public_models_and_custom_refs(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        ModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings()).list_models()

    public = [model for model in response.models if model.origin == "public"]
    assert len(public) == 5
    assert {model.provider for model in public} == {"huggingface"}
    assert all(model.access_policy in {"open", "gated"} for model in public)
    assert all(model.anatomy_coverage for model in public)
    assert [model.model_ref for model in response.models if model.origin == "custom"] == [
        "xreport:checkpoint_epoch_48",
    ]
    assert set(response.providers) == {"huggingface", "xreport"}
    assert response.providers["huggingface"].status == "not_installed"
    assert response.providers["xreport"].status == "ready"
    assert response.models[0].status == "not_installed"
    assert response.models[0].installation_state == "not_installed"
    assert response.models[0].available_actions == ["download"]
    xreport = response.models[-1]
    assert xreport.output_sections == ["raw_report"]

###############################################################################
def test_catalog_disables_huggingface_when_local_only_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        ModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings(hf_local_only=False)).list_models()

    public = [model for model in response.models if model.origin == "public"]
    assert len(public) == 5
    assert all(model.status == "disabled" for model in public)
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

    cxrmate_multi = next(model for model in response.models if model.model_ref.endswith("aehrc/cxrmate-multi-tf"))
    assert cxrmate_multi.model_revision == "330721b9aa5bba201a3eb88eba4dd9a6607f3e7a"
    assert cxrmate_multi.model_loader == "auto_model"
    assert cxrmate_multi.processor_loader == "auto"
    assert cxrmate_multi.adapter == "cxrmate_multi"
    assert cxrmate_multi.output_sections == ["findings", "impression"]
    assert cxrmate_multi.max_current_images == 16
    assert cxrmate_multi.license == "Apache-2.0"
    assert cxrmate_multi.hardware_demand == "low"
    assert cxrmate_multi.status == "not_installed"

###############################################################################
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
