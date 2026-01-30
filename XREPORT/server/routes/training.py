from __future__ import annotations

import os
import shutil
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status
import sqlalchemy

from XREPORT.server.schemas.training import (
    CheckpointInfo,
    CheckpointsResponse,
    CheckpointMetadataResponse,
    DeleteResponse,
    StartTrainingRequest,
    ResumeTrainingRequest,
    TrainingStatusResponse,
)
from XREPORT.server.schemas.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.jobs import JobManager, job_manager
from XREPORT.server.utils.repository.serializer import (
    DataSerializer,
    ModelSerializer,
    CHECKPOINT_PATH,
)
from XREPORT.server.utils.configurations.server import server_settings
from XREPORT.server.database.database import database
from XREPORT.server.utils.constants import CHECKPOINTS_SUMMARY_TABLE
from XREPORT.server.utils.constants import TRAINING_DATASET_TABLE
from XREPORT.server.utils.learning.training.worker import (
    ProcessWorker,
    run_resume_training_process,
    run_training_process,
)


###############################################################################
class TrainingState:
    """Encapsulates all training session state."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {
            "is_training": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "loss": 0.0,
            "val_loss": 0.0,
            "accuracy": 0.0,
            "val_accuracy": 0.0,
            "progress_percent": 0,
            "elapsed_seconds": 0,
            "chart_data": [],
            "epoch_boundaries": [],
            "available_metrics": [],
        }
        self.worker: ProcessWorker | None = None
        self.current_job_id: str | None = None

    # -----------------------------------------------------------------------------
    def update_metrics(self, message: dict[str, Any]) -> None:
        """Update state from a training callback message."""
        self.state.update({
            "current_epoch": message.get("epoch", 0),
            "total_epochs": message.get("total_epochs", 0),
            "loss": message.get("loss", 0.0),
            "val_loss": message.get("val_loss", 0.0),
            "accuracy": message.get("accuracy", 0.0),
            "val_accuracy": message.get("val_accuracy", 0.0),
            "progress_percent": message.get("progress_percent", 0),
            "elapsed_seconds": message.get("elapsed_seconds", 0),
        })
        if message.get("type") == "training_plot":
            chart_data = message.get("chart_data")
            chart_point = message.get("chart_point")
            if isinstance(chart_data, list):
                self.state["chart_data"] = chart_data
            elif isinstance(chart_point, dict):
                self.state["chart_data"].append(chart_point)

            epoch_boundaries = message.get("epoch_boundaries")
            epoch_boundary = message.get("epoch_boundary")
            if isinstance(epoch_boundaries, list):
                self.state["epoch_boundaries"] = epoch_boundaries
            elif isinstance(epoch_boundary, (int, float)):
                self.state["epoch_boundaries"].append(epoch_boundary)

            self.state["available_metrics"] = message.get("metrics", [])

    # -----------------------------------------------------------------------------
    def reset_for_new_session(self, total_epochs: int, job_id: str) -> None:
        """Reset state for a new training session."""
        self.state["is_training"] = True
        self.state["current_epoch"] = 0
        self.state["total_epochs"] = total_epochs
        self.state["loss"] = 0.0
        self.state["val_loss"] = 0.0
        self.state["accuracy"] = 0.0
        self.state["val_accuracy"] = 0.0
        self.state["progress_percent"] = 0
        self.state["elapsed_seconds"] = 0
        self.state["chart_data"] = []
        self.state["epoch_boundaries"] = []
        self.state["available_metrics"] = []
        self.current_job_id = job_id

    # -----------------------------------------------------------------------------
    def finish_session(self) -> None:
        """Mark training session as complete."""
        self.state["is_training"] = False
        self.worker = None
        self.current_job_id = None


training_state = TrainingState()


# -----------------------------------------------------------------------------
def handle_training_progress(job_id: str, message: dict[str, Any]) -> None:
    training_state.update_metrics(message)

    if not job_id:
        return

    message_type = message.get("type")
    if message_type == "training_update":
        job_manager.update_progress(job_id, float(message.get("progress_percent", 0)))
        job_manager.update_result(
            job_id,
            {
                "current_epoch": message.get("epoch", 0),
                "total_epochs": message.get("total_epochs", 0),
                "loss": message.get("loss", 0.0),
                "val_loss": message.get("val_loss", 0.0),
                "accuracy": message.get("accuracy", 0.0),
                "val_accuracy": message.get("val_accuracy", 0.0),
                "progress_percent": message.get("progress_percent", 0),
                "elapsed_seconds": message.get("elapsed_seconds", 0),
            },
        )
    elif message_type == "training_plot":
        job_manager.update_result(
            job_id,
            {
                "chart_data": training_state.state["chart_data"],
                "epoch_boundaries": training_state.state["epoch_boundaries"],
                "available_metrics": training_state.state["available_metrics"],
            },
        )


# -----------------------------------------------------------------------------
def drain_worker_progress(job_id: str, worker: ProcessWorker) -> None:
    while True:
        message = worker.poll(timeout=0.0)
        if message is None:
            return
        handle_training_progress(job_id, message)


# -----------------------------------------------------------------------------
def monitor_training_process(
    job_id: str,
    worker: ProcessWorker,
    stop_timeout_seconds: float,
) -> dict[str, Any]:
    stop_requested_at: float | None = None

    while worker.is_alive():
        if job_manager.should_stop(job_id) and not worker.is_interrupted():
            worker.stop()
            stop_requested_at = time.monotonic()
        if stop_requested_at is not None:
            elapsed = time.monotonic() - stop_requested_at
            if elapsed >= stop_timeout_seconds:
                worker.terminate()
                break
        message = worker.poll(timeout=0.25)
        if message is not None:
            handle_training_progress(job_id, message)
            drain_worker_progress(job_id, worker)

    worker.join(timeout=5)
    drain_worker_progress(job_id, worker)

    result_payload = worker.read_result()
    if result_payload is None:
        if worker.exitcode not in (0, None) and not job_manager.should_stop(job_id):
            raise RuntimeError(
                f"Training process exited with code {worker.exitcode}"
            )
        return {}
    if "error" in result_payload and result_payload["error"]:
        raise RuntimeError(str(result_payload["error"]))
    if "result" in result_payload:
        return result_payload["result"] or {}
    return {}


# -----------------------------------------------------------------------------
def run_training_job(
    configuration: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Blocking training function that runs in background thread."""
    worker = ProcessWorker()
    training_state.worker = worker
    try:
        worker.start(
            target=run_training_process,
            kwargs={"configuration": configuration},
        )

        return monitor_training_process(
            job_id,
            worker,
            stop_timeout_seconds=5.0,
        )
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        worker.cleanup()
        training_state.finish_session()


# -----------------------------------------------------------------------------
def run_resume_training_job(
    checkpoint: str,
    additional_epochs: int,
    job_id: str,
) -> dict[str, Any]:
    """Blocking resume training function that runs in background thread."""
    worker = ProcessWorker()
    training_state.worker = worker
    try:
        worker.start(
            target=run_resume_training_process,
            kwargs={
                "checkpoint": checkpoint,
                "additional_epochs": additional_epochs,
            },
        )

        return monitor_training_process(
            job_id,
            worker,
            stop_timeout_seconds=5.0,
        )
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        worker.cleanup()
        training_state.finish_session()


###############################################################################
class TrainingEndpoint:    

    JOB_TYPE = "training"
    CHECKPOINT_EMPTY_MESSAGE = "Checkpoint name cannot be empty"
    NO_TRAINING_DATA_MESSAGE = "No training data found. Please process a dataset first."

    def __init__(
        self,
        router: APIRouter,
        job_manager: JobManager,
        training_state: TrainingState,
    ) -> None:
        self.router = router
        self.job_manager = job_manager
        self.training_state = training_state

    # -----------------------------------------------------------------------------
    def get_checkpoints(self) -> CheckpointsResponse:
        """Get list of available checkpoints (JSON config only, no model loading)."""
        modser = ModelSerializer()
        checkpoint_names = modser.scan_checkpoints_folder()
        
        checkpoints = []
        for name in checkpoint_names:
            try:
                # Only load JSON configuration files, NOT the model
                checkpoint_path = os.path.join(CHECKPOINT_PATH, name)
                _, _, session = modser.load_training_configuration(checkpoint_path)
                checkpoints.append(CheckpointInfo(
                    name=name,
                    epochs=session.get("epochs", 0),
                    loss=session.get("history", {}).get("loss", [0])[-1] if session.get("history") else 0.0,
                    val_loss=session.get("history", {}).get("val_loss", [0])[-1] if session.get("history") else 0.0,
                ))
            except Exception as e:
                logger.warning(f"Failed to load checkpoint config {name}: {e}")
                checkpoints.append(CheckpointInfo(name=name, epochs=0, loss=0.0, val_loss=0.0))
        
        return CheckpointsResponse(checkpoints=checkpoints)

    # -------------------------------------------------------------------------
    def get_checkpoint_metadata(self, checkpoint: str) -> CheckpointMetadataResponse:
        checkpoint = checkpoint.strip()
        if not checkpoint:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=self.CHECKPOINT_EMPTY_MESSAGE,
            )

        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint)
        if not os.path.isdir(checkpoint_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint not found: {checkpoint}",
            )

        try:
            modser = ModelSerializer()
            configuration, metadata, session = modser.load_training_configuration(checkpoint_path)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load checkpoint metadata: {exc}",
            ) from exc

        return CheckpointMetadataResponse(
            checkpoint=checkpoint,
            configuration=configuration,
            metadata=metadata,
            session=session,
        )

    # -----------------------------------------------------------------------------
    def delete_checkpoint(self, checkpoint: str) -> DeleteResponse:
        checkpoint = checkpoint.strip()
        if not checkpoint:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=self.CHECKPOINT_EMPTY_MESSAGE,
            )

        if self.training_state.state.get("is_training"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete checkpoints while training is active",
            )

        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint)
        if not os.path.isdir(checkpoint_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint not found: {checkpoint}",
            )

        try:
            shutil.rmtree(checkpoint_path)
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete checkpoint: {exc}",
            ) from exc

        with database.backend.engine.begin() as conn:
            inspector = sqlalchemy.inspect(conn)
            if inspector.has_table(CHECKPOINTS_SUMMARY_TABLE):
                conn.execute(
                    sqlalchemy.text(
                        'DELETE FROM "CHECKPOINTS_SUMMARY" WHERE checkpoint = :checkpoint'
                    ),
                    {"checkpoint": checkpoint},
                )

        return DeleteResponse(
            success=True,
            message=f"Deleted checkpoint {checkpoint}",
        )

    # -----------------------------------------------------------------------------
    def get_training_status(self) -> TrainingStatusResponse:
        return TrainingStatusResponse(
            job_id=self.training_state.current_job_id,
            is_training=self.training_state.state["is_training"],
            current_epoch=self.training_state.state["current_epoch"],
            total_epochs=self.training_state.state["total_epochs"],
            loss=self.training_state.state["loss"],
            val_loss=self.training_state.state["val_loss"],
            accuracy=self.training_state.state["accuracy"],
            val_accuracy=self.training_state.state["val_accuracy"],
            progress_percent=self.training_state.state["progress_percent"],
            elapsed_seconds=self.training_state.state["elapsed_seconds"],
            poll_interval=server_settings.training.update_frequency_seconds,
        )

    # -----------------------------------------------------------------------------
    def start_training(self, request: StartTrainingRequest) -> JobStartResponse:
        if self.job_manager.is_job_running("training"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Training is already in progress",
            )
        
        serializer = DataSerializer()
        
        # Build configuration from request
        configuration = request.model_dump()
        configuration.pop("sample_size", None)
        
        # Inject configurations from configurations.json
        configuration["use_mixed_precision"] = server_settings.training.use_mixed_precision
        configuration["training_seed"] = server_settings.global_settings.seed
        configuration["dataloader_workers"] = server_settings.training.dataloader_workers
        configuration["prefetch_factor"] = server_settings.training.prefetch_factor
        configuration["pin_memory"] = server_settings.training.pin_memory
        configuration["persistent_workers"] = server_settings.training.persistent_workers
        configuration["update_frequency_seconds"] = server_settings.training.update_frequency_seconds
        
        stored_metadata = serializer.load_training_data(only_metadata=True)
        if not stored_metadata:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=self.NO_TRAINING_DATA_MESSAGE,
            )
        if serializer.count_rows(TRAINING_DATASET_TABLE) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=self.NO_TRAINING_DATA_MESSAGE,
            )
        
        # Start background job
        job_id = self.job_manager.start_job(
            job_type="training",
            runner=run_training_job,
            kwargs={
                "configuration": configuration,
            },
        )
        
        self.training_state.reset_for_new_session(configuration.get("epochs", 10), job_id)
        self.job_manager.update_result(
            job_id,
            {
                "current_epoch": self.training_state.state["current_epoch"],
                "total_epochs": self.training_state.state["total_epochs"],
                "loss": self.training_state.state["loss"],
                "val_loss": self.training_state.state["val_loss"],
                "accuracy": self.training_state.state["accuracy"],
                "val_accuracy": self.training_state.state["val_accuracy"],
                "progress_percent": self.training_state.state["progress_percent"],
                "elapsed_seconds": self.training_state.state["elapsed_seconds"],
                "chart_data": self.training_state.state["chart_data"],
                "epoch_boundaries": self.training_state.state["epoch_boundaries"],
                "available_metrics": self.training_state.state["available_metrics"],
            },
        )

        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize training job",
            )
        
        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message="Training job started",
            poll_interval=server_settings.training.update_frequency_seconds,
        )

    # -----------------------------------------------------------------------------
    def resume_training(self, request: ResumeTrainingRequest) -> JobStartResponse:
        if self.job_manager.is_job_running("training"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Training is already in progress",
            )
        
        # Initialize serializers
        serializer = DataSerializer()
        modser = ModelSerializer()
        
        stored_metadata = serializer.load_training_data(only_metadata=True)
        if not stored_metadata:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=self.NO_TRAINING_DATA_MESSAGE,
            )

        checkpoint = request.checkpoint.strip()
        if not checkpoint:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=self.CHECKPOINT_EMPTY_MESSAGE,
            )

        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint)
        if not os.path.isdir(checkpoint_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint not found: {checkpoint}",
            )

        try:
            _, _, session = modser.load_training_configuration(checkpoint_path)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load checkpoint metadata: {exc}",
            ) from exc

        from_epoch = session.get("epochs", 0)
        
        # Start background job
        job_id = self.job_manager.start_job(
            job_type="training",
            runner=run_resume_training_job,
            kwargs={
                "checkpoint": checkpoint,
                "additional_epochs": request.additional_epochs,
            },
        )
        
        # Update training state
        self.training_state.reset_for_new_session(from_epoch + request.additional_epochs, job_id)
        self.training_state.state["current_epoch"] = from_epoch
        self.job_manager.update_result(
            job_id,
            {
                "current_epoch": self.training_state.state["current_epoch"],
                "total_epochs": self.training_state.state["total_epochs"],
                "loss": self.training_state.state["loss"],
                "val_loss": self.training_state.state["val_loss"],
                "accuracy": self.training_state.state["accuracy"],
                "val_accuracy": self.training_state.state["val_accuracy"],
                "progress_percent": self.training_state.state["progress_percent"],
                "elapsed_seconds": self.training_state.state["elapsed_seconds"],
                "chart_data": self.training_state.state["chart_data"],
                "epoch_boundaries": self.training_state.state["epoch_boundaries"],
                "available_metrics": self.training_state.state["available_metrics"],
            },
        )

        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize training resume job",
            )
        
        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message=f"Training resumed from epoch {from_epoch}",
            poll_interval=server_settings.training.update_frequency_seconds,
        )

    # -----------------------------------------------------------------------------
    def get_training_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return JobStatusResponse(**job_status)

    # -----------------------------------------------------------------------------
    def cancel_training_job(self, job_id: str) -> JobCancelResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        
        if self.training_state.worker is not None:
            self.training_state.worker.stop()
        
        success = self.job_manager.cancel_job(job_id)
        
        if success:
            logger.info("Training stop requested for job %s", job_id)
        
        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )

    # -----------------------------------------------------------------------------
    def stop_training(self) -> TrainingStatusResponse:
        """Legacy stop endpoint - maintained for backward compatibility."""
        if not self.training_state.state["is_training"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No training is currently in progress",
            )
        
        if self.training_state.worker is not None:
            self.training_state.worker.stop()
            logger.info("Training stop requested")
        
        # Also cancel via job manager if we have a job_id
        if self.training_state.current_job_id:
            self.job_manager.cancel_job(self.training_state.current_job_id)
        
        return TrainingStatusResponse(
            job_id=self.training_state.current_job_id,
            is_training=self.training_state.state["is_training"],
            current_epoch=self.training_state.state["current_epoch"],
            total_epochs=self.training_state.state["total_epochs"],
            loss=self.training_state.state["loss"],
            val_loss=self.training_state.state["val_loss"],
            accuracy=self.training_state.state["accuracy"],
            val_accuracy=self.training_state.state["val_accuracy"],
            progress_percent=self.training_state.state["progress_percent"],
            elapsed_seconds=self.training_state.state["elapsed_seconds"],
        )

    # -----------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all training-related routes."""
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
            "/checkpoints/{checkpoint}",
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
        self.router.add_api_route(
            "/stop",
            self.stop_training,
            methods=["POST"],
            response_model=TrainingStatusResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
router = APIRouter(prefix="/training", tags=["training"])
training_endpoint = TrainingEndpoint(
    router=router,
    job_manager=job_manager,
    training_state=training_state,
)
training_endpoint.add_routes()
