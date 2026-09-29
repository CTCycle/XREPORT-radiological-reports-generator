from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from server.api.errors import register_service_error_handlers
from server.api.inference import InferenceEndpoint
from server.configurations import ServerSettings
from server.services.errors import ConflictError
from server.services.inference import InferenceImageStore, InferenceService
from server.services.inference_catalog import InferenceModelCatalog
from server.services.jobs import JobManager
from server.services.model_installation import ModelInstallationManager


def test_generation_rejects_degraded_public_models_before_starting_a_job() -> None:
    model_ref = "huggingface:future/degraded-model"
    model = SimpleNamespace(
        model_ref=model_ref,
        origin="public",
        validation_status="degraded",
        status="ready",
    )
    catalog = cast(
        InferenceModelCatalog,
        SimpleNamespace(list_models=lambda: SimpleNamespace(models=[model])),
    )
    job_manager = MagicMock(spec=JobManager)
    service = InferenceService(
        job_manager=job_manager,
        inference_image_store=InferenceImageStore(),
        server_settings=cast(ServerSettings, SimpleNamespace()),
        model_catalog=catalog,
        installation_manager=cast(ModelInstallationManager, MagicMock()),
        repository=MagicMock(),
    )

    with pytest.raises(
        ConflictError,
        match="Model failed qualification and cannot be used for inference",
    ):
        service.generate_reports(
            model_ref=model_ref,
            generation_profile="deterministic",
            clinical_context="",
            images=[],
        )

    job_manager.start_job.assert_not_called()


def test_inference_api_returns_conflict_for_a_degraded_public_model() -> None:
    model_ref = "huggingface:future/degraded-model"
    model = SimpleNamespace(
        model_ref=model_ref,
        origin="public",
        validation_status="degraded",
        status="ready",
    )
    catalog = cast(
        InferenceModelCatalog,
        SimpleNamespace(list_models=lambda: SimpleNamespace(models=[model])),
    )
    job_manager = MagicMock(spec=JobManager)
    service = InferenceService(
        job_manager=job_manager,
        inference_image_store=InferenceImageStore(),
        server_settings=cast(ServerSettings, SimpleNamespace()),
        model_catalog=catalog,
        installation_manager=cast(ModelInstallationManager, MagicMock()),
        repository=MagicMock(),
    )
    application = FastAPI()
    register_service_error_handlers(application)
    router = InferenceEndpoint(router=APIRouter(prefix="/inference"), service=service)
    router.add_routes()
    application.include_router(router.router)

    with TestClient(application) as client:
        response = client.post(
            "/inference/generate",
            data={
                "model_ref": model_ref,
                "generation_profile": "deterministic",
                "clinical_context": "",
            },
            files={"images": ("scan.png", b"image", "image/png")},
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Model failed qualification and cannot be used for inference: "
        f"{model_ref}"
    }
    job_manager.start_job.assert_not_called()
