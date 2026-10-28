from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Response, status

from xreport_studio.domain.model_architecture import (
    ArchitecturePreset,
    ArchitecturePresetRequest,
    ArchitecturePresetsResponse,
    ArchitecturePreviewRequest,
    ArchitecturePreviewResponse,
    CheckpointArchitectureResponse,
    EncodersResponse,
)

if TYPE_CHECKING:
    from xreport_studio.services.model_architecture import ModelArchitectureService

###############################################################################
class ModelArchitectureEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        router: APIRouter,
        service: ModelArchitectureService | None = None,
    ) -> None:
        self.router = router
        self._service = service

    # -------------------------------------------------------------------------
    @property
    def service(self) -> ModelArchitectureService:
        if self._service is None:
            from xreport_studio.services.model_architecture import (
                get_model_architecture_service,
            )

            self._service = get_model_architecture_service()
        return self._service

    # -------------------------------------------------------------------------
    def get_encoders(self) -> EncodersResponse:
        return self.service.get_encoders()

    # -------------------------------------------------------------------------
    def list_presets(self) -> ArchitecturePresetsResponse:
        return self.service.list_presets()

    # -------------------------------------------------------------------------
    def create_preset(self, request: ArchitecturePresetRequest) -> ArchitecturePreset:
        return self.service.create_preset(request)

    # -------------------------------------------------------------------------
    def update_preset(
        self, key: str, request: ArchitecturePresetRequest
    ) -> ArchitecturePreset:
        return self.service.update_preset(key, request)

    # -------------------------------------------------------------------------
    def delete_preset(self, key: str) -> Response:
        self.service.delete_preset(key)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # -------------------------------------------------------------------------
    def restore_presets(self) -> ArchitecturePresetsResponse:
        return self.service.restore_builtin_presets()

    # -------------------------------------------------------------------------
    def preview(self, request: ArchitecturePreviewRequest) -> ArchitecturePreviewResponse:
        return self.service.preview(request)

    # -------------------------------------------------------------------------
    def get_checkpoint_architecture(
        self, checkpoint: str
    ) -> CheckpointArchitectureResponse:
        return self.service.get_checkpoint_architecture(checkpoint)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/encoders",
            self.get_encoders,
            methods=["GET"],
            response_model=EncodersResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/presets",
            self.list_presets,
            methods=["GET"],
            response_model=ArchitecturePresetsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/presets",
            self.create_preset,
            methods=["POST"],
            response_model=ArchitecturePreset,
            status_code=status.HTTP_201_CREATED,
        )
        self.router.add_api_route(
            "/presets/restore",
            self.restore_presets,
            methods=["POST"],
            response_model=ArchitecturePresetsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/presets/{key}",
            self.update_preset,
            methods=["PUT"],
            response_model=ArchitecturePreset,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/presets/{key}",
            self.delete_preset,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
            response_class=Response,
        )
        self.router.add_api_route(
            "/preview",
            self.preview,
            methods=["POST"],
            response_model=ArchitecturePreviewResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/checkpoints/{checkpoint}/architecture",
            self.get_checkpoint_architecture,
            methods=["GET"],
            response_model=CheckpointArchitectureResponse,
            status_code=status.HTTP_200_OK,
        )

###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/model", tags=["model"])
    ModelArchitectureEndpoint(router=router).add_routes()
    return router


router = get_router()
