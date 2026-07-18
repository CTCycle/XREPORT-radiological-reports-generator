from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile, status

from server.domain.inference import GenerationProfile, InferenceImage, InferenceModelsResponse
from server.domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from server.services.inference import InferenceService, get_inference_service

class InferenceEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        router: APIRouter,
        service: InferenceService | None = None,
    ) -> None:
        self.router = router
        self.service = get_inference_service() if service is None else service

    # -------------------------------------------------------------------------
    def get_models(self) -> InferenceModelsResponse:
        return self.service.get_models()

    # -------------------------------------------------------------------------
    async def generate_reports(
        self,
        model_ref: str = Form(...),
        generation_profile: GenerationProfile = Form(...),
        clinical_context: str = Form(...),
        images: list[UploadFile] = File(...),
    ) -> JobStartResponse:
        parsed_images: list[InferenceImage] = []
        for image in images:
            filename = (image.filename or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
            content = await image.read()
            parsed_images.append(
                InferenceImage(
                    filename=filename,
                    content_type=image.content_type or "",
                    data=content,
                    size_bytes=len(content),
                )
            )
        return self.service.generate_reports(
            model_ref=model_ref,
            generation_profile=generation_profile,
            clinical_context=clinical_context,
            images=parsed_images,
        )

    # -------------------------------------------------------------------------
    def get_inference_job_status(self, job_id: str) -> JobStatusResponse:
        return self.service.get_inference_job_status(job_id)

    # -------------------------------------------------------------------------
    def cancel_inference_job(self, job_id: str) -> JobCancelResponse:
        return self.service.cancel_inference_job(job_id)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/models",
            self.get_models,
            methods=["GET"],
            response_model=InferenceModelsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/generate",
            self.generate_reports,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_inference_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_inference_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )

###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/inference", tags=["inference"])
    InferenceEndpoint(router=router).add_routes()
    return router


router = get_router()
