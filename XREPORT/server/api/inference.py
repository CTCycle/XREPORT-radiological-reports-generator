from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile, status

from XREPORT.server.domain.inference import CheckpointsResponse
from XREPORT.server.domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from XREPORT.server.services.inference import inference_service


###############################################################################
class InferenceEndpoint:
    def __init__(self, router: APIRouter) -> None:
        self.router = router

    def get_checkpoints(self) -> CheckpointsResponse:
        return inference_service.get_checkpoints()

    async def generate_reports(
        self,
        checkpoint: str = Form(...),
        generation_mode: str = Form(...),
        images: list[UploadFile] = File(...),
    ) -> JobStartResponse:
        return await inference_service.generate_reports(
            checkpoint=checkpoint,
            generation_mode=generation_mode,
            images=images,
        )

    def get_inference_job_status(self, job_id: str) -> JobStatusResponse:
        return inference_service.get_inference_job_status(job_id)

    def cancel_inference_job(self, job_id: str) -> JobCancelResponse:
        return inference_service.cancel_inference_job(job_id)

    def add_routes(self) -> None:
        self.router.add_api_route(
            "/checkpoints",
            self.get_checkpoints,
            methods=["GET"],
            response_model=CheckpointsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/generate",
            self.generate_reports,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_inference_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_inference_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
router = APIRouter(prefix="/inference", tags=["inference"])
inference_endpoint = InferenceEndpoint(router=router)
inference_endpoint.add_routes()
