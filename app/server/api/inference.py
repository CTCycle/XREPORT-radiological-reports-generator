from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from server.domain.inference import (
    GenerationProfile,
    InferenceImage,
    InferenceGenerateRequest,
    InferenceModelsResponse,
    ModelMaintenanceRequest,
    ModelUpdateCheckRequest,
    ModelUpdateCheckResponse,
)
from server.domain.jobs import JobStartResponse
if TYPE_CHECKING:
    from server.services.inference import InferenceService

###############################################################################
def parse_generation_request(
    model_ref: str = Form(...),
    generation_profile: GenerationProfile = Form(...),
    clinical_context: str = Form(""),
) -> InferenceGenerateRequest:
    return InferenceGenerateRequest(
        model_ref=model_ref,
        generation_profile=generation_profile,
        clinical_context=clinical_context,
    )

###############################################################################
class InferenceEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        router: APIRouter,
        service: InferenceService | None = None,
    ) -> None:
        self.router = router
        self._service = service

    # -------------------------------------------------------------------------
    @property
    def service(self) -> InferenceService:
        if self._service is None:
            from server.services.inference import get_inference_service

            self._service = get_inference_service()
        return self._service

    # -------------------------------------------------------------------------
    def get_models(self) -> InferenceModelsResponse:
        return self.service.get_models()

    # -------------------------------------------------------------------------
    def check_model_update(self, request: ModelUpdateCheckRequest) -> ModelUpdateCheckResponse:
        return self.service.get_model_update(request.model_ref)

    # -------------------------------------------------------------------------
    def maintain_model(self, request: ModelMaintenanceRequest) -> JobStartResponse:
        return self.service.start_model_maintenance(
            model_ref=request.model_ref,
            action=request.action,
            revision=request.revision,
        )

    # -------------------------------------------------------------------------
    async def generate_reports(
        self,
        request: Annotated[
            InferenceGenerateRequest,
            Depends(parse_generation_request),
        ],
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
            model_ref=request.model_ref,
            generation_profile=request.generation_profile,
            clinical_context=request.clinical_context,
            images=parsed_images,
        )

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
            "/models/check-update",
            self.check_model_update,
            methods=["POST"],
            response_model=ModelUpdateCheckResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/models/maintenance",
            self.maintain_model,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/generate",
            self.generate_reports,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/inference", tags=["inference"])
    InferenceEndpoint(router=router).add_routes()
    return router


router = get_router()
