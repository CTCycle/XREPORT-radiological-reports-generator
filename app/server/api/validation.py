from __future__ import annotations

from fastapi import APIRouter, status

from server.domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from server.domain.validation import (
    CheckpointEvaluationReportResponse,
    CheckpointEvaluationRequest,
    ValidationReportResponse,
    ValidationRequest,
)
from server.services.validation_runs import ValidationService, get_validation_service

###############################################################################
class ValidationEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        router: APIRouter,
        service: ValidationService | None = None,
    ) -> None:
        self.router = router
        self.service = get_validation_service() if service is None else service

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
    async def get_validation_report(self, dataset_name: str) -> ValidationReportResponse:
        return await self.service.get_validation_report(dataset_name)

    # -------------------------------------------------------------------------
    async def get_validation_job_status(self, job_id: str) -> JobStatusResponse:
        return await self.service.get_validation_job_status(job_id)

    # -------------------------------------------------------------------------
    async def cancel_validation_job(self, job_id: str) -> JobCancelResponse:
        return await self.service.cancel_validation_job(job_id)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
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
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_validation_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_validation_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )

###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/validation", tags=["validation"])
    ValidationEndpoint(router=router).add_routes()
    return router


router = get_router()
