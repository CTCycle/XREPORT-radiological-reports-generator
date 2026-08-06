from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from server.domain.inference import InferenceImage, ProviderGenerationResult
from server.domain.jobs import JobState
from server.services.inference import InferenceImageStore, run_inference_job
import server.services.inference as inference_service
from server.services.jobs import JobManager

###############################################################################
def _image() -> InferenceImage:
    return InferenceImage(
        filename="scan.png",
        content_type="image/png",
        data=b"fixture",
        size_bytes=7,
    )

###############################################################################
def _setup_job(
    manager: JobManager,
    image_store: InferenceImageStore,
    *,
    job_id: str,
    request_id: str,
) -> None:
    manager.jobs[job_id] = JobState(
        job_id=job_id,
        job_type="inference",
        status="running",
    )
    image_store.store(request_id, [_image()])
    image_store.link_job(job_id, request_id)

###############################################################################
def _patch_runtime(monkeypatch, manager, image_store, provider, repository) -> None:
    monkeypatch.setattr(inference_service, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        inference_service,
        "get_inference_image_store",
        lambda: image_store,
    )

    ###############################################################################
    class RuntimeStub:

        # -------------------------------------------------------------------------
        def generate(self, **kwargs):
            generation = provider.generate(**kwargs)
            return ProviderGenerationResult(
                reports=generation.reports,
                display_sections=generation.display_sections,
                metadata=generation.metadata,
                provenance=generation.provenance,
            )

    monkeypatch.setattr(inference_service, "get_inference_runtime", lambda: RuntimeStub())
    monkeypatch.setattr(inference_service, "InferenceRepository", lambda: repository)

###############################################################################
def test_cancelled_after_generation_does_not_persist_partial_reports(monkeypatch) -> None:
    manager = JobManager()
    image_store = InferenceImageStore()
    job_id = "cancel-after-generation"
    request_id = "request-cancelled"
    _setup_job(manager, image_store, job_id=job_id, request_id=request_id)

    ###############################################################################
    class CancellingProvider:

        # -------------------------------------------------------------------------
        def generate(self, **_kwargs):
            manager.cancel_job(job_id)
            return ProviderGenerationResult(
                reports={"scan.png": "Partial report"},
                display_sections={"scan.png": {"raw_report": "Partial report"}},
                metadata=[],
                provenance={"provider": "huggingface"},
            )

    repository = MagicMock()
    _patch_runtime(monkeypatch, manager, image_store, CancellingProvider(), repository)

    result = run_inference_job(
        model_ref="huggingface:test/model",
        model_revision="a" * 40,
        model_manifest={"revision": "a" * 40},
        generation_profile="deterministic",
        clinical_context="",
        request_id=request_id,
        job_id=job_id,
    )

    assert result["reports"] == {}
    repository.save_generated_reports.assert_not_called()
    assert image_store.get(request_id) is None

###############################################################################
def test_timeout_does_not_persist_reports(monkeypatch) -> None:
    manager = JobManager()
    image_store = InferenceImageStore()
    job_id = "timeout-no-persistence"
    request_id = "request-timeout"
    _setup_job(manager, image_store, job_id=job_id, request_id=request_id)
    provider = MagicMock()
    provider.generate.side_effect = TimeoutError("inference deadline exceeded")
    repository = MagicMock()
    _patch_runtime(monkeypatch, manager, image_store, provider, repository)

    with pytest.raises(TimeoutError, match="deadline exceeded"):
        run_inference_job(
            model_ref="huggingface:test/model",
            model_revision="a" * 40,
            model_manifest={"revision": "a" * 40},
            generation_profile="deterministic",
            clinical_context="",
            request_id=request_id,
            job_id=job_id,
        )

    repository.save_generated_reports.assert_not_called()
    assert image_store.get(request_id) is None
