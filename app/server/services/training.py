from __future__ import annotations

import shutil
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from server.domain.training import (
    CheckpointInfo,
    CheckpointsResponse,
    CheckpointMetadataResponse,
    DeleteResponse,
    StartTrainingRequest,
    ResumeTrainingRequest,
    TrainingStatusResponse,
)
from server.domain.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from server.common.utils.logger import logger
from server.common.utils.security import (
    resolve_checkpoint_path,
    validate_checkpoint_name,
)
from server.services.jobs import JobManager, get_job_manager
from server.repositories.serialization.data import DataSerializer
from server.repositories.serialization.model import ModelSerializer
from server.common.path import CHECKPOINT_PATH
from server.configurations.startup import get_server_settings
from server.learning.training.worker import (
    ProcessWorker,
    run_resume_training_process,
    run_training_process,
)

###############################################################################
class TrainingState:
    """Encapsulates all training session state."""

    def __init__(self) -> None:
        self.state = self.build_state(is_training=False, total_epochs=0)
        self.worker: ProcessWorker | None = None
        self.current_job_id: str | None = None

    # -----------------------------------------------------------------------------
    def build_state(self, is_training: bool, total_epochs: int) -> dict[str, Any]:
        return {
            "is_training": is_training,
            "current_epoch": 0,
            "total_epochs": total_epochs,
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

    # -----------------------------------------------------------------------------
    def result_payload(self) -> dict[str, Any]:
        return {
            "current_epoch": self.state["current_epoch"],
            "total_epochs": self.state["total_epochs"],
            "loss": self.state["loss"],
            "val_loss": self.state["val_loss"],
            "accuracy": self.state["accuracy"],
            "val_accuracy": self.state["val_accuracy"],
            "progress_percent": self.state["progress_percent"],
            "elapsed_seconds": self.state["elapsed_seconds"],
            "chart_data": list(self.state["chart_data"]),
            "epoch_boundaries": list(self.state["epoch_boundaries"]),
            "available_metrics": list(self.state["available_metrics"]),
        }

    # -----------------------------------------------------------------------------
    def update_metrics(self, message: dict[str, Any]) -> None:
        """Update state from a training callback message."""
        self.state.update(
            {
                "current_epoch": message.get("epoch", 0),
                "total_epochs": message.get("total_epochs", 0),
                "loss": message.get("loss", 0.0),
                "val_loss": message.get("val_loss", 0.0),
                "accuracy": message.get("accuracy", 0.0),
                "val_accuracy": message.get("val_accuracy", 0.0),
                "progress_percent": message.get("progress_percent", 0),
                "elapsed_seconds": message.get("elapsed_seconds", 0),
            }
        )
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
        self.state = self.build_state(is_training=True, total_epochs=total_epochs)
        self.current_job_id = job_id

    # -----------------------------------------------------------------------------
    def finish_session(self) -> None:
        """Mark training session as complete."""
        self.state["is_training"] = False
        self.worker = None
        self.current_job_id = None


@lru_cache(maxsize=1)
def get_training_state() -> TrainingState:
    return TrainingState()


# -----------------------------------------------------------------------------
def handle_training_progress(job_id: str, message: dict[str, Any]) -> None:
    training_state = get_training_state()
    training_state.update_metrics(message)

    if not job_id:
        return

    message_type = message.get("type")
    if message_type == "training_update":
        get_job_manager().update_progress(job_id, float(message.get("progress_percent", 0)))
        get_job_manager().update_result(
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
        get_job_manager().update_result(
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
def request_worker_stop_if_needed(
    job_id: str,
    worker: ProcessWorker,
    stop_requested_at: float | None,
) -> float | None:
    if not get_job_manager().should_stop(job_id):
        return stop_requested_at

    if stop_requested_at is None:
        stop_requested_at = time.monotonic()

    if not worker.is_interrupted():
        worker.stop()

    return stop_requested_at


# -----------------------------------------------------------------------------
def enforce_worker_stop_timeout(
    job_id: str,
    worker: ProcessWorker,
    stop_requested_at: float | None,
    stop_timeout_seconds: float,
) -> bool:
    if stop_requested_at is None:
        return False

    elapsed = time.monotonic() - stop_requested_at
    if elapsed < stop_timeout_seconds:
        return False

    logger.warning(
        "Training job %s did not stop within %.2fs, forcing termination",
        job_id,
        stop_timeout_seconds,
    )
    worker.terminate()
    return True


# -----------------------------------------------------------------------------
def read_worker_result(job_id: str, worker: ProcessWorker) -> dict[str, Any]:
    result_payload = worker.read_result()
    if result_payload is None:
        if worker.exitcode not in (0, None) and not get_job_manager().should_stop(job_id):
            raise RuntimeError(f"Training process exited with code {worker.exitcode}")
        return {}

    if "error" in result_payload and result_payload["error"]:
        raise RuntimeError(str(result_payload["error"]))

    if "result" in result_payload:
        return result_payload["result"] or {}

    return {}


# -----------------------------------------------------------------------------
def monitor_training_process(
    job_id: str,
    worker: ProcessWorker,
    stop_timeout_seconds: float,
) -> dict[str, Any]:
    stop_requested_at: float | None = None

    while worker.is_alive():
        stop_requested_at = request_worker_stop_if_needed(
            job_id=job_id,
            worker=worker,
            stop_requested_at=stop_requested_at,
        )
        if enforce_worker_stop_timeout(
            job_id=job_id,
            worker=worker,
            stop_requested_at=stop_requested_at,
            stop_timeout_seconds=stop_timeout_seconds,
        ):
            break

        message = worker.poll(timeout=0.25)
        if message is not None:
            handle_training_progress(job_id, message)
            drain_worker_progress(job_id, worker)

    worker.join(timeout=5)
    drain_worker_progress(job_id, worker)

    return read_worker_result(job_id=job_id, worker=worker)


# -----------------------------------------------------------------------------
def run_training_job(
    configuration: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Blocking training function that runs in background thread."""
    training_state = get_training_state()
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
    training_state = get_training_state()
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
class TrainingService:
    JOB_TYPE = "training"
    CHECKPOINT_EMPTY_MESSAGE = "Checkpoint name cannot be empty"
    NO_TRAINING_DATA_MESSAGE = "No training data found. Please process a dataset first."

    def __init__(
        self,
        job_manager: JobManager,
        training_state: TrainingState,
    ) -> None:
        self.job_manager = job_manager
        self.training_state = training_state

    # -----------------------------------------------------------------------------
    def apply_runtime_training_configuration(
        self, configuration: dict[str, Any]
    ) -> None:
        server_settings = get_server_settings()
        configuration["training_seed"] = server_settings.global_settings.seed
        configuration["polling_interval"] = server_settings.jobs.polling_interval

    # -----------------------------------------------------------------------------
    def initialize_training_state(
        self, job_id: str, total_epochs: int, current_epoch: int = 0
    ) -> None:
        self.training_state.reset_for_new_session(total_epochs, job_id)
        self.training_state.state["current_epoch"] = current_epoch
        self.job_manager.update_result(job_id, self.training_state.result_payload())

    # -----------------------------------------------------------------------------
    def build_job_start_response(
        self,
        job_id: str,
        message: str,
        initialization_error: str,
    ) -> JobStartResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=initialization_error,
            )

        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message=message,
            poll_interval=get_server_settings().jobs.polling_interval,
        )

    # -----------------------------------------------------------------------------
    def get_checkpoints(self) -> CheckpointsResponse:
        """Get list of available checkpoints (JSON config only, no model loading)."""
        modser = ModelSerializer()
        checkpoint_names = modser.scan_checkpoints_folder()

        checkpoints = []
        for name in checkpoint_names:
            try:
                # Only load JSON configuration files, NOT the model
                checkpoint_path = CHECKPOINT_PATH / name
                _, _, session = modser.load_training_configuration(checkpoint_path)
                checkpoints.append(
                    CheckpointInfo(
                        name=name,
                        epochs=session.get("epochs", 0),
                        loss=session.get("history", {}).get("loss", [0])[-1]
                        if session.get("history")
                        else 0.0,
                        val_loss=session.get("history", {}).get("val_loss", [0])[-1]
                        if session.get("history")
                        else 0.0,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint config {name}: {e}")
                checkpoints.append(
                    CheckpointInfo(name=name, epochs=0, loss=0.0, val_loss=0.0)
                )

        return CheckpointsResponse(checkpoints=checkpoints)

    # -------------------------------------------------------------------------
    def get_checkpoint_metadata(self, checkpoint: str) -> CheckpointMetadataResponse:
        try:
            checkpoint = validate_checkpoint_name(checkpoint)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        try:
            checkpoint_path = resolve_checkpoint_path(checkpoint)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint not found: {checkpoint}",
            )

        try:
            modser = ModelSerializer()
            configuration, metadata, session = modser.load_training_configuration(
                checkpoint_path
            )
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
        try:
            checkpoint = validate_checkpoint_name(checkpoint)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        if self.training_state.state.get("is_training"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete checkpoints while training is active",
            )

        try:
            checkpoint_path = resolve_checkpoint_path(checkpoint)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint not found: {checkpoint}",
            )

        try:
            shutil.rmtree(checkpoint_path_obj)
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete checkpoint: {exc}",
            ) from exc

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
            poll_interval=get_server_settings().jobs.polling_interval,
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

        self.apply_runtime_training_configuration(configuration)

        dataset_name = configuration.get("dataset_name")
        stored_metadata = serializer.load_training_data(
            only_metadata=True,
            dataset_name=dataset_name,
        )
        if not stored_metadata:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=self.NO_TRAINING_DATA_MESSAGE,
            )
        train_data, validation_data, _ = serializer.load_training_data(
            dataset_name=dataset_name
        )
        if train_data.empty and validation_data.empty:
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

        self.initialize_training_state(
            job_id=job_id,
            total_epochs=configuration.get("epochs", 10),
        )

        return self.build_job_start_response(
            job_id=job_id,
            message="Training job started",
            initialization_error="Failed to initialize training job",
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

        try:
            checkpoint = validate_checkpoint_name(request.checkpoint)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        try:
            checkpoint_path = resolve_checkpoint_path(checkpoint)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint not found: {checkpoint}",
            )

        try:
            _, _, session = modser.load_training_configuration(checkpoint_path_obj)
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

        self.initialize_training_state(
            job_id=job_id,
            total_epochs=from_epoch + request.additional_epochs,
            current_epoch=from_epoch,
        )

        return self.build_job_start_response(
            job_id=job_id,
            message=f"Training resumed from epoch {from_epoch}",
            initialization_error="Failed to initialize training resume job",
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

###############################################################################
@lru_cache(maxsize=1)
def get_training_service() -> TrainingService:
    return TrainingService(
        job_manager=get_job_manager(),
        training_state=get_training_state(),
    )

