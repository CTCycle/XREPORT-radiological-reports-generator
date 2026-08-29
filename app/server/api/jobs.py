from __future__ import annotations

from fastapi import APIRouter, Query, status

from server.domain.jobs import JobCancelResponse, JobListResponse, JobStatusResponse
from server.services.errors import NotFoundError
from server.services.jobs import JobManager, get_job_manager


class JobsEndpoint:
    def __init__(self, router: APIRouter, job_manager: JobManager | None = None) -> None:
        self.router = router
        self.job_manager = job_manager or get_job_manager()

    # -------------------------------------------------------------------------
    def list_jobs(
        self,
        job_type: str | None = Query(default=None),
        job_status: str | None = Query(default=None, alias="status"),
    ) -> JobListResponse:
        return JobListResponse(
            jobs=[
                JobStatusResponse(**job)
                for job in self.job_manager.list_jobs(
                    job_type=job_type,
                    status=job_status,
                )
            ]
        )

    # -------------------------------------------------------------------------
    def get_job(self, job_id: str) -> JobStatusResponse:
        job = self.job_manager.get_job_status(job_id)
        if job is None:
            raise NotFoundError(detail=f"Job not found: {job_id}")
        return JobStatusResponse(**job)

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> JobCancelResponse:
        if self.job_manager.get_job_status(job_id) is None:
            raise NotFoundError(detail=f"Job not found: {job_id}")
        success = self.job_manager.cancel_job(job_id)
        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message=(
                "Cancellation requested" if success else "Job cannot be cancelled"
            ),
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "",
            self.list_jobs,
            methods=["GET"],
            response_model=JobListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/{job_id}",
            self.get_job,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/{job_id}",
            self.cancel_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )


def get_router() -> APIRouter:
    router = APIRouter(prefix="/jobs", tags=["jobs"])
    JobsEndpoint(router=router).add_routes()
    return router


router = get_router()
