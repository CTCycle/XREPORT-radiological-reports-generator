from __future__ import annotations

from fastapi import APIRouter, status

from XREPORT.server.domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from XREPORT.server.domain.validation import (
    CheckpointEvaluationReportResponse,
    CheckpointEvaluationRequest,
    ValidationReportResponse,
    ValidationRequest,
)
from XREPORT.server.services.validation_runs import validation_service


###############################################################################
class ValidationEndpoint:
    def __init__(self, router: APIRouter) -> None:
        self.router = router

    async def run_validation(self, request: ValidationRequest) -> JobStartResponse:
        return await validation_service.run_validation(request)

    async def evaluate_checkpoint(
        self, request: CheckpointEvaluationRequest
    ) -> JobStartResponse:
        return await validation_service.evaluate_checkpoint(request)

    async def get_checkpoint_evaluation_report(
        self, checkpoint: str
    ) -> CheckpointEvaluationReportResponse:
        return await validation_service.get_checkpoint_evaluation_report(checkpoint)

    async def get_validation_report(self, dataset_name: str) -> ValidationReportResponse:
        return await validation_service.get_validation_report(dataset_name)

    async def get_validation_job_status(self, job_id: str) -> JobStatusResponse:
        return await validation_service.get_validation_job_status(job_id)

    async def cancel_validation_job(self, job_id: str) -> JobCancelResponse:
        return await validation_service.cancel_validation_job(job_id)

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
router = APIRouter(prefix="/validation", tags=["validation"])
validation_endpoint = ValidationEndpoint(router=router)
validation_endpoint.add_routes()
