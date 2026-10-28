from __future__ import annotations

from pathlib import Path
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import FileResponse

from xreport_shared.common.path import CHECKPOINTS_DIR
from xreport_shared.domain.checkpoint_exchange import EXCHANGE_FILE_EXTENSION, ExportRequest
from xreport_shared.domain.jobs import JobStartResponse
from xreport_shared.repositories.checkpoint_archive import MAX_TOTAL_BYTES
from xreport_shared.services.errors import PayloadTooLargeError

if TYPE_CHECKING:
    from xreport_shared.services.checkpoint_exchange import CheckpointExchangeService

UPLOAD_CHUNK_BYTES = 1024 * 1024
STAGING_UPLOAD_PREFIX = ".import-upload-"

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
            from xreport_shared.services.checkpoint_exchange import (
                get_checkpoint_exchange_service,
            )

            self._service = get_checkpoint_exchange_service()
        return self._service

    # -------------------------------------------------------------------------
    async def import_checkpoint(
        self,
        file: UploadFile = File(...),
        rename_to: str | None = Form(default=None),
    ) -> JobStartResponse:
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        staged = CHECKPOINTS_DIR / f"{STAGING_UPLOAD_PREFIX}{uuid.uuid4().hex}{EXCHANGE_FILE_EXTENSION}"
        written = 0
        try:
            with staged.open("xb") as target:
                while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if written > MAX_TOTAL_BYTES:
                        raise PayloadTooLargeError("Checkpoint archive is too large")
                    target.write(chunk)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        return self.service.start_import(
            staged, (rename_to or "").strip() or None
        )

    # -------------------------------------------------------------------------
    def export_checkpoint(self, request: ExportRequest) -> JobStartResponse:
        return self.service.start_export(request.checkpoint, request.include_encoder)

    # -------------------------------------------------------------------------
    def download_export(self, filename: str) -> FileResponse:
        path: Path = self.service.resolve_export_path(filename)
        return FileResponse(
            path, media_type="application/zip", filename=path.name
        )

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
