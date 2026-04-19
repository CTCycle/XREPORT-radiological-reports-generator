from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, status

from XREPORT.server.domain.training import DatasetUploadResponse
from XREPORT.server.services.upload import UploadService, get_upload_state


###############################################################################
class UploadEndpoint:
    """Endpoint for dataset upload operations."""

    def __init__(self, router: APIRouter, upload_service: UploadService) -> None:
        self.router = router
        self.upload_service = upload_service

    # -----------------------------------------------------------------------------
    async def upload_dataset(
        self, file: UploadFile = File(...)
    ) -> DatasetUploadResponse:
        return await self.upload_service.upload_dataset(file)

    # -----------------------------------------------------------------------------
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
router = APIRouter(prefix="/upload", tags=["upload"])
upload_endpoint = UploadEndpoint(
    router=router,
    upload_service=UploadService(get_upload_state()),
)
upload_endpoint.add_routes()
