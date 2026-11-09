from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import FileResponse

from xreport_shared.domain.checkpoint_exchange import ExportRequest
from xreport_shared.domain.jobs import JobStartResponse
from xreport_shared.services.checkpoint_exchange import (
    CheckpointExchangeService,
    get_checkpoint_exchange_service,
)


###############################################################################
class CheckpointExchangeEndpoint:
    # -------------------------------------------------------------------------
    def __init__(
        self, router: APIRouter, service: CheckpointExchangeService | None = None
    ) -> None:
        self.router = router
        self._service = service

    # -------------------------------------------------------------------------
    @property
    def service(self) -> CheckpointExchangeService:
        if self._service is None:
            self._service = get_checkpoint_exchange_service()
        return self._service

    # -------------------------------------------------------------------------
    async def import_checkpoint(
        self,
        file: UploadFile = File(...),
        rename_to: str | None = Form(default=None),
    ) -> JobStartResponse:
        staged = await self.service.stage_upload(file.read)
        return self.service.start_import(staged, (rename_to or "").strip() or None)

    # -------------------------------------------------------------------------
    def export_checkpoint(self, request: ExportRequest) -> JobStartResponse:
        return self.service.start_export(request.checkpoint, request.include_encoder)

    # -------------------------------------------------------------------------
    def download_export(self, filename: str) -> FileResponse:
        path: Path = self.service.resolve_export_path(filename)
        return FileResponse(path, media_type="application/zip", filename=path.name)

    # -------------------------------------------------------------------------
    def add_import_routes(self) -> None:
        self.router.add_api_route(
            "/import",
            self.import_checkpoint,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )

    # -------------------------------------------------------------------------
    def add_export_routes(self) -> None:
        self.router.add_api_route(
            "/export",
            self.export_checkpoint,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/exports/{filename}",
            self.download_export,
            methods=["GET"],
            response_model=None,
        )


###############################################################################
def get_import_router() -> APIRouter:
    router = APIRouter(prefix="/exchange", tags=["checkpoint-exchange"])
    CheckpointExchangeEndpoint(router=router).add_import_routes()
    return router


###############################################################################
def get_export_router() -> APIRouter:
    router = APIRouter(prefix="/exchange", tags=["checkpoint-exchange"])
    CheckpointExchangeEndpoint(router=router).add_export_routes()
    return router
