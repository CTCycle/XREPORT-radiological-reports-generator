from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, File, UploadFile, status

from server.domain.training import DatasetUploadResponse
if TYPE_CHECKING:
    from server.services.upload import UploadService

###############################################################################
class UploadEndpoint:
    """Endpoint for dataset upload operations."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        router: APIRouter,
        upload_service: UploadService | None = None,
    ) -> None:
        self.router = router
        self._upload_service = upload_service

    # -------------------------------------------------------------------------
    @property
    def upload_service(self) -> UploadService:
        if self._upload_service is None:
            from server.services.upload import UploadService, get_upload_state

            self._upload_service = UploadService(get_upload_state())
        return self._upload_service

    # -------------------------------------------------------------------------
    async def upload_dataset(
        self, file: UploadFile = File(...)
    ) -> DatasetUploadResponse:
        contents = await file.read()
        return self.upload_service.upload_dataset(
            filename=file.filename or "",
            contents=contents,
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all upload-related routes."""
        self.router.add_api_route(
            "/dataset",
            self.upload_dataset,
            methods=["POST"],
            response_model=DatasetUploadResponse,
            status_code=status.HTTP_200_OK,
        )

###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/upload", tags=["upload"])
    UploadEndpoint(router=router).add_routes()
    return router


router = get_router()
