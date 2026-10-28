from __future__ import annotations

from fastapi import APIRouter, status

from xreport_shared.domain.jobs import JobStartResponse
from xreport_inference.domain.storage_settings import (
    StorageApplyRequest,
    StoragePathRequest,
    StorageStatus,
    StorageValidationResponse,
)
from xreport_inference.services.storage_settings import StorageSettingsService


###############################################################################
class StorageSettingsEndpoint:

    # -------------------------------------------------------------------------
    def __init__(self, service: StorageSettingsService | None = None) -> None:
        self._service = service

    # -------------------------------------------------------------------------
    @property
    def service(self) -> StorageSettingsService:
        if self._service is None:
            self._service = StorageSettingsService()
        return self._service

    # -------------------------------------------------------------------------
    def get_status(self) -> StorageStatus:
        return self.service.get_status()

    # -------------------------------------------------------------------------
    def validate(self, payload: StoragePathRequest) -> StorageValidationResponse:
        return self.service.validate(payload.path)

    # -------------------------------------------------------------------------
    def apply(self, payload: StorageApplyRequest) -> JobStartResponse:
        return self.service.apply(payload.path, payload.mode)

    # -------------------------------------------------------------------------
    def add_routes(self, router: APIRouter) -> None:
        router.add_api_route(
            "",
            self.get_status,
            methods=["GET"],
            response_model=StorageStatus,
            status_code=status.HTTP_200_OK,
        )
        router.add_api_route(
            "/validate",
            self.validate,
            methods=["POST"],
            response_model=StorageValidationResponse,
            status_code=status.HTTP_200_OK,
        )
        router.add_api_route(
            "/apply",
            self.apply,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )


###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/settings/storage", tags=["storage-settings"])
    StorageSettingsEndpoint().add_routes(router)
    return router


router = get_router()

__all__ = ["StorageSettingsEndpoint", "router"]
