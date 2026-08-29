from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from server.common.path import ROOT_DIR
from server.configurations import InferenceSettings
from server.services.inference_catalog import InferenceModelCatalog
from server.services.model_installation import ModelInstallationManager

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
def _catalog(checkpoints: list[object]) -> InferenceModelCatalog:
    repository = SimpleNamespace(list_checkpoints=lambda: checkpoints)
    return InferenceModelCatalog(_settings(), checkpoint_repository=repository)

###############################################################################
def _checkpoint(name: str = "checkpoint_epoch_48", *, complete: bool = True) -> object:
    return SimpleNamespace(
        name=name,
        name_key=name.casefold(),
        path=Path("app/resources/models/checkpoints") / name,
        artifact_complete=complete,
    )

###############################################################################
def test_catalog_exposes_available_public_and_custom_model_sources() -> None:
    response = _catalog([_checkpoint()]).list_models()

    public = [model for model in response.models if model.origin == "public"]
    custom = [model for model in response.models if model.origin == "custom"]
    assert public
    assert all(model.provider == "huggingface" for model in public)
    assert [model.model_ref for model in custom] == ["xreport:checkpoint_epoch_48"]
    assert set(response.providers) == {"huggingface", "xreport"}
    assert response.providers["huggingface"].status == "not_installed"
    assert response.providers["xreport"].status == "ready"
    assert all(model.available_actions == ["download"] for model in public)

###############################################################################
def test_catalog_disables_public_models_when_huggingface_runtime_is_disabled() -> None:
    catalog = _catalog([_checkpoint()])
    catalog.settings = _settings(hf_local_only=False)
    response = catalog.list_models()

    public = [model for model in response.models if model.origin == "public"]
    assert public
    assert all(model.status == "disabled" for model in public)
    assert response.providers["huggingface"].status == "disabled"

###############################################################################
def test_catalog_hides_xreport_provider_without_registered_checkpoints() -> None:
    response = _catalog([]).list_models()

    assert not any(model.provider == "xreport" for model in response.models)
    assert response.providers["xreport"].status == "not_installed"

###############################################################################
def test_catalog_marks_verified_active_installation_ready(monkeypatch) -> None:
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

    model = _catalog([]).list_models().models[0]

    assert model.status == "ready"
    assert model.installation_state == "active"
    assert model.integrity_status == "verified"
    assert model.local_path == "app/resources/models/huggingface/installed/active"
