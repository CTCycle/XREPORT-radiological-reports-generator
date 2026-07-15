from __future__ import annotations

from fastapi import APIRouter

from server.api.inference import InferenceEndpoint
from server.domain.inference import InferenceImage
from server.domain.inference import InferenceModelsResponse
from server.domain.jobs import JobStartResponse
from tests.conftest import run_async_in_thread

###############################################################################
class UploadFileStub:

    # -------------------------------------------------------------------------
    def __init__(self, filename: str, content_type: str, content: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._content = content

    # -------------------------------------------------------------------------
    async def read(self) -> bytes:
        return self._content

###############################################################################
class InferenceServiceStub:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.captured: dict[str, object] | None = None

    # -------------------------------------------------------------------------
    def get_models(self):
        return InferenceModelsResponse(models=[], providers={})

    # -------------------------------------------------------------------------
    def get_inference_job_status(self, job_id: str):
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def cancel_inference_job(self, job_id: str):
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def generate_reports(
        self,
        model_ref: str,
        generation_profile: str,
        clinical_context: str,
        images: list[InferenceImage],
    ) -> JobStartResponse:
        self.captured = {
            "model_ref": model_ref,
            "generation_profile": generation_profile,
            "clinical_context": clinical_context,
            "images": images,
        }
        return JobStartResponse(
            job_id="job-1",
            job_type="inference",
            status="pending",
            message=f"Inference job started for {len(images)} images",
            poll_interval=1.0,
        )

###############################################################################
def test_inference_endpoint_converts_uploads_to_domain_images() -> None:
    service = InferenceServiceStub()
    endpoint = InferenceEndpoint(router=APIRouter(), service=service)
    uploads = [
        UploadFileStub(filename="scan-1.png", content_type="image/png", content=b"a"),
        UploadFileStub(filename="nested\\scan-2.jpg", content_type="image/jpeg", content=b"bb"),
    ]

    response = run_async_in_thread(
        endpoint.generate_reports(
            model_ref="xreport:checkpoint_a",
            generation_profile="deterministic",
            clinical_context="",
            images=uploads,
        )
    )

    assert response == JobStartResponse(
        job_id="job-1",
        job_type="inference",
        status="pending",
        message="Inference job started for 2 images",
        poll_interval=1.0,
    )
    assert service.captured is not None
    assert service.captured["model_ref"] == "xreport:checkpoint_a"
    assert service.captured["generation_profile"] == "deterministic"
    assert service.captured["clinical_context"] == ""
    assert service.captured["images"] == [
        InferenceImage(
            filename="scan-1.png",
            content_type="image/png",
            data=b"a",
            size_bytes=1,
        ),
        InferenceImage(
            filename="scan-2.jpg",
            content_type="image/jpeg",
            data=b"bb",
            size_bytes=2,
        ),
    ]


def test_inference_endpoint_exposes_model_catalog() -> None:
    endpoint = InferenceEndpoint(router=APIRouter(), service=InferenceServiceStub())

    assert endpoint.get_models() == InferenceModelsResponse(models=[], providers={})
