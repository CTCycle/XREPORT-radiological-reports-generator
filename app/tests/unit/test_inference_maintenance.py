from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from server.domain.inference import InferenceModelsResponse, ModelAvailability, ModelCapabilities
from server.services.errors import NotFoundError
from server.services.inference import InferenceImageStore, InferenceService

###############################################################################
def test_unknown_model_reference_is_not_a_maintenance_target(monkeypatch: pytest.MonkeyPatch) -> None:
    service = InferenceService(
        job_manager=MagicMock(),
        inference_image_store=InferenceImageStore(),
        server_settings=MagicMock(),
        model_catalog=MagicMock(),
        installation_manager=MagicMock(),
        runtime=MagicMock(),
        repository=MagicMock(),
    )
    monkeypatch.setattr(
        service,
        "get_models",
        lambda: InferenceModelsResponse(models=[], providers={}),
    )

    with pytest.raises(NotFoundError, match="not in the local inference catalog"):
        service.start_model_maintenance(
            model_ref="huggingface:example/unknown-model",
            action="delete_local",
            revision=None,
        )

###############################################################################
def test_model_maintenance_starts_for_download_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    service = InferenceService(
        job_manager=MagicMock(),
        inference_image_store=InferenceImageStore(),
        server_settings=SimpleNamespace(jobs=SimpleNamespace(polling_interval=2)),
        model_catalog=MagicMock(),
        installation_manager=MagicMock(),
        runtime=MagicMock(),
        repository=MagicMock(),
    )
    model = ModelAvailability(
        model_ref="huggingface:example/model",
        provider="huggingface",
        display_name="Example model",
        description="Example",
        status="not_installed",
        category="research",
        input_semantics="single_image",
        capabilities=ModelCapabilities(),
        model_revision="a" * 40,
    )
    monkeypatch.setattr(
        service,
        "get_models",
        lambda: InferenceModelsResponse(models=[model], providers={}),
    )
    service.job_manager.start_job.return_value = "job-1"
    service.job_manager.get_job_status.return_value = {
        "job_id": "job-1",
        "job_type": "model_maintenance",
        "status": "pending",
    }

    result = service.start_model_maintenance(
        model_ref=model.model_ref,
        action="download",
        revision="b" * 40,
    )

    assert result.job_id == "job-1"
    kwargs = service.job_manager.start_job.call_args.kwargs["kwargs"]
    assert kwargs["model_ref"] == model.model_ref
    assert kwargs["revision"] == "b" * 40
