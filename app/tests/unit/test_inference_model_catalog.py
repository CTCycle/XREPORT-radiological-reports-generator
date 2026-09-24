from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from server.common.path import ROOT_DIR
from server.configurations import InferenceSettings
from server.services.inference_catalog import InferenceModelCatalog
from server.services.model_installation import ModelInstallationManager


###############################################################################
def _settings(*, hf_local_only: bool = True) -> InferenceSettings:
    return InferenceSettings(
        hf_local_only=hf_local_only,
        device="auto",
        model_timeout=600,
    )


###############################################################################
class _InstallationManager(ModelInstallationManager):
    def __init__(
        self,
        inspector: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> None:
        self.inspector = inspector

    def inspect(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        return self.inspector(manifest)


###############################################################################
def _not_installed() -> dict[str, Any]:
    return {
        "metadata": {},
        "state": "not_installed",
        "integrity": "unknown",
        "active_revision": None,
        "active_path": None,
        "candidate": None,
        "candidate_path": None,
        "candidate_revision": None,
    }


###############################################################################
def _catalog(
    checkpoints: list[object],
    inspect_installation: Callable[[Mapping[str, Any]], dict[str, Any]] = (
        lambda _manifest: _not_installed()
    ),
) -> InferenceModelCatalog:
    repository = SimpleNamespace(list_checkpoints=lambda: checkpoints)
    installation_manager = _InstallationManager(inspect_installation)
    return InferenceModelCatalog(
        _settings(),
        installation_manager=installation_manager,
        checkpoint_repository=repository,
    )


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
def test_catalog_exposes_chexone_findings_only_contract() -> None:
    chexone = next(
        model
        for model in _catalog([]).list_models().models
        if model.model_ref == "huggingface:StanfordAIMI/CheXOne"
    )

    assert chexone.output_sections == ["findings"]
    assert chexone.capabilities.findings is True
    assert chexone.capabilities.impression is False
    assert chexone.provider == "huggingface"
    assert chexone.origin == "public"
    assert chexone.adapter == "chexone"


###############################################################################
def test_catalog_marks_verified_active_installation_ready() -> None:
    active_path = (
        ROOT_DIR
        / "app"
        / "resources"
        / "models"
        / "huggingface"
        / "installed"
        / "active"
    )
    response = _catalog(
        [],
        inspect_installation=lambda manifest: {
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

    model = response.list_models().models[0]

    assert model.status == "ready"
    assert model.installation_state == "active"
    assert model.integrity_status == "verified"
    assert model.local_path == "app/resources/models/huggingface/installed/active"
