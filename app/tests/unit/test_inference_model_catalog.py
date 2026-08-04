from __future__ import annotations

import json
from pathlib import Path

from server.configurations import InferenceSettings
from server.domain.inference import InferenceManifest
from server.services.inference_catalog import InferenceModelCatalog
from server.services import inference_catalog


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
def test_catalog_lists_all_configured_models_and_xreport_refs(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        ModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings()).list_models()

    assert [model.model_ref for model in response.models] == [
        "huggingface:google/medgemma-1.5-4b-it",
        "huggingface:microsoft/maira-2",
        "huggingface:erjui/CheXagent-2-3b-srrg-impression",
        "huggingface:aehrc/cxrmate-2",
        "huggingface:nathansutton/generate-cxr",
        "xreport:checkpoint_epoch_48",
    ]
    assert "ollama" not in response.providers
    assert response.providers["huggingface"].status == "not_installed"
    assert response.providers["xreport"].status == "ready"
    assert [model.status for model in response.models[:5]] == [
        "gated", "incompatible", "disabled", "disabled", "not_installed"
    ]
    xreport = response.models[-1]
    assert xreport.output_sections == ["raw_report"]

###############################################################################
def test_catalog_disables_huggingface_when_local_only_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.services.inference_catalog.ModelSerializer",
        ModelSerializerStub,
    )

    response = InferenceModelCatalog(_settings(hf_local_only=False)).list_models()

    medgemma = next(model for model in response.models if model.model_ref.endswith("google/medgemma-1.5-4b-it"))
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
        lambda self, repository_id, revision, **kwargs: repository_id == "google/medgemma-1.5-4b-it" and revision == REVISION,
    )

    response = InferenceModelCatalog(_settings()).list_models()

    medgemma = next(model for model in response.models if model.model_ref.endswith("google/medgemma-1.5-4b-it"))
    assert medgemma.model_revision == REVISION
    assert medgemma.model_loader == "image_text_to_text"
    assert medgemma.processor_loader == "auto"
    assert medgemma.adapter == "medgemma"
    assert medgemma.output_sections == ["raw_report"]
    assert medgemma.max_current_images == 1
    assert medgemma.license == "Health AI Developer Foundations terms of use"
    assert medgemma.status == "gated"
    assert "gated" in (medgemma.status_message or "").lower()

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


def test_catalog_requires_exact_contract_receipt_for_ready(monkeypatch, tmp_path: Path) -> None:
    payload = json.loads(inference_catalog.CATALOG_PATH.read_text(encoding="utf-8"))
    nathan = next(model for model in payload["models"] if model["adapter"] == "generate_cxr_blip")
    nathan["validation_status"] = "pending"
    nathan["validation_message"] = None
    manifest_path = tmp_path / "inference_models.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(inference_catalog, "CATALOG_PATH", manifest_path)
    monkeypatch.setattr(inference_catalog, "VALIDATION_RECEIPTS_DIR", tmp_path)
    monkeypatch.setattr(inference_catalog.ModelSerializer, "scan_checkpoints_folder", lambda self: [])
    monkeypatch.setattr(
        inference_catalog.HuggingFaceProvider,
        "is_cached",
        lambda self, repository_id, revision, **kwargs: repository_id == "nathansutton/generate-cxr",
    )

    entry = next(
        model for model in InferenceManifest.model_validate(payload).models
        if model.adapter == "generate_cxr_blip"
    )
    receipt = {
        "status": "passed",
        "real_inference": True,
        "model_ref": entry.model_ref,
        "revision": entry.revision,
        "contract_hash": inference_catalog.validation_contract_hash(entry),
        "reports": {"scan.png": "A report."},
        "display_sections": {"scan.png": {"raw_report": "A report."}},
    }
    (tmp_path / f"nathansutton__generate-cxr-{entry.revision}.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    response = InferenceModelCatalog(_settings()).list_models()
    model = next(model for model in response.models if model.adapter == "generate_cxr_blip")
    assert model.status == "ready"

    receipt["contract_hash"] = "stale"
    (tmp_path / f"nathansutton__generate-cxr-{entry.revision}.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    stale_response = InferenceModelCatalog(_settings()).list_models()
    stale_model = next(model for model in stale_response.models if model.adapter == "generate_cxr_blip")
    assert stale_model.status == "unvalidated"
