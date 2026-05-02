from __future__ import annotations

from fastapi import APIRouter, status

from server.domain.training import (
    CheckpointMetadataResponse,
    CheckpointsResponse,
    DeleteResponse,
    ResumeTrainingRequest,
    StartTrainingRequest,
    TrainingStatusResponse,
)
from server.domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)
from server.services.training import TrainingService, get_training_service


###############################################################################
class TrainingEndpoint:
    def __init__(
        self,
        router: APIRouter,
        service: TrainingService | None = None,
    ) -> None:
        self.router = router
        self.service = get_training_service() if service is None else service

    def get_checkpoints(self) -> CheckpointsResponse:
        return self.service.get_checkpoints()

    def get_checkpoint_metadata(self, checkpoint: str) -> CheckpointMetadataResponse:
        return self.service.get_checkpoint_metadata(checkpoint)

    def delete_checkpoint(self, checkpoint: str) -> DeleteResponse:
        return self.service.delete_checkpoint(checkpoint)

    def get_training_status(self) -> TrainingStatusResponse:
        return self.service.get_training_status()

    def start_training(self, request: StartTrainingRequest) -> JobStartResponse:
        return self.service.start_training(request)

    def resume_training(self, request: ResumeTrainingRequest) -> JobStartResponse:
        return self.service.resume_training(request)

    def get_training_job_status(self, job_id: str) -> JobStatusResponse:
        return self.service.get_training_job_status(job_id)

    def cancel_training_job(self, job_id: str) -> JobCancelResponse:
        return self.service.cancel_training_job(job_id)

    def add_routes(self) -> None:
        self.router.add_api_route(
            "/checkpoints",
            self.get_checkpoints,
            methods=["GET"],
            response_model=CheckpointsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/checkpoints/{checkpoint}/metadata",
            self.get_checkpoint_metadata,
            methods=["GET"],
            response_model=CheckpointMetadataResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/checkpoints/{checkpoint:path}",
            self.delete_checkpoint,
            methods=["DELETE"],
            response_model=DeleteResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/status",
            self.get_training_status,
            methods=["GET"],
            response_model=TrainingStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/start",
            self.start_training,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/resume",
            self.resume_training,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_training_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_training_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/training", tags=["training"])
    TrainingEndpoint(router=router).add_routes()
    return router


router = get_router()
