from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, Query, status

from server.domain.jobs import JobStartResponse
from server.domain.validation import (
    CheckpointEvaluationReportResponse,
    CheckpointEvaluationRequest,
    RecordStatisticsResponse,
    ValidationReportResponse,
    ValidationRequest,
    ValidationRunListResponse,
)

if TYPE_CHECKING:
    from server.services.validation_runs import ValidationService

###############################################################################
class ValidationEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        router: APIRouter,
        service: ValidationService | None = None,
    ) -> None:
        self.router = router
        self._service = service

    # -------------------------------------------------------------------------
    @property
    def service(self) -> ValidationService:
        if self._service is None:
            from server.services.validation_runs import get_validation_service

            self._service = get_validation_service()
        return self._service

    # -------------------------------------------------------------------------
    async def run_validation(self, request: ValidationRequest) -> JobStartResponse:
        return await self.service.run_validation(request)

    # -------------------------------------------------------------------------
    async def evaluate_checkpoint(
        self, request: CheckpointEvaluationRequest
    ) -> JobStartResponse:
        return await self.service.evaluate_checkpoint(request)

    # -------------------------------------------------------------------------
    async def get_checkpoint_evaluation_report(
        self, checkpoint: str
    ) -> CheckpointEvaluationReportResponse:
        return await self.service.get_checkpoint_evaluation_report(checkpoint)

    # -------------------------------------------------------------------------
    async def get_validation_report(
        self, dataset_name: str
    ) -> ValidationReportResponse:
        return await self.service.get_validation_report(dataset_name)

    # -------------------------------------------------------------------------
    async def list_validation_runs(self, dataset_name: str) -> ValidationRunListResponse:
        return await self.service.list_validation_runs(dataset_name)

    # -------------------------------------------------------------------------
    async def get_validation_run(
        self, dataset_name: str, run_id: int
    ) -> ValidationReportResponse:
        return await self.service.get_validation_report(dataset_name, run_id)

    # -------------------------------------------------------------------------
    async def list_run_records(
        self,
        dataset_name: str,
        run_id: int,
        kind: Literal["text", "image"] = Query(..., description="Statistics domain"),
        sort: str | None = Query(default=None, max_length=40),
        descending: bool = False,
        search: str | None = Query(default=None, max_length=128),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> RecordStatisticsResponse:
        return await self.service.list_run_records(
            dataset_name, run_id, kind, sort, descending, search, limit, offset
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/runs/{dataset_name}",
            self.list_validation_runs,
            methods=["GET"],
            response_model=ValidationRunListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/runs/{dataset_name}/{run_id}",
            self.get_validation_run,
            methods=["GET"],
            response_model=ValidationReportResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/runs/{dataset_name}/{run_id}/records",
            self.list_run_records,
            methods=["GET"],
            response_model=RecordStatisticsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/run",
            self.run_validation,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/checkpoint",
            self.evaluate_checkpoint,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/checkpoint/reports/{checkpoint}",
            self.get_checkpoint_evaluation_report,
            methods=["GET"],
            response_model=CheckpointEvaluationReportResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/reports/{dataset_name}",
            self.get_validation_report,
            methods=["GET"],
            response_model=ValidationReportResponse,
            status_code=status.HTTP_200_OK,
        )

###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/validation", tags=["validation"])
    ValidationEndpoint(router=router).add_routes()
    return router


router = get_router()
