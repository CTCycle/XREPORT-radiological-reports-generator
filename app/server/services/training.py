from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from server.services.errors import (
    BadRequestError,
    ConflictError,
    InternalServiceError,
    NotFoundError,
)

from server.domain.training import (
    CheckpointInfo,
    CheckpointsResponse,
    CheckpointMetadataResponse,
    DeleteResponse,
    StartTrainingRequest,
    ResumeTrainingRequest,
)
from server.domain.jobs import (
    JobStartResponse,
)
from server.common.utils.logger import logger
from server.common.utils.security import (
    validate_checkpoint_name,
)
from server.services.jobs import JobExecutionError, JobManager, get_job_manager
from server.repositories.serialization.dataset import DatasetRepository
from server.repositories.serialization.model import ModelSerializer
from server.repositories.checkpoints import (
    CheckpointReferencedError,
    CheckpointRegistryError,
    CheckpointRepository,
)
from server.configurations.startup import get_server_settings
from server.services.training_worker import (
    ProcessWorker,
    run_resume_training_process,
    run_training_process,
)


###############################################################################
class TrainingRuntime:
    """Owns only the internal worker handle for the active training job."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.worker: ProcessWorker | None = None


###############################################################################
@lru_cache(maxsize=1)
def get_training_runtime() -> TrainingRuntime:
    return TrainingRuntime()


###############################################################################
def handle_training_progress(job_id: str, message: dict[str, Any]) -> None:
    if not job_id:
        return

    manager = get_job_manager()
    message_type = message.get("type")
    if message_type == "training_update":
        manager.update_progress(job_id, float(message.get("progress_percent", 0)))
        manager.update_result(
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
        current = manager.get_job_status(job_id) or {}
        existing = current.get("result") or {}
        chart_data = message.get("chart_data")
        if not isinstance(chart_data, list):
            chart_data = list(existing.get("chart_data") or [])
            chart_point = message.get("chart_point")
            if isinstance(chart_point, dict):
                chart_data.append(chart_point)
        epoch_boundaries = message.get("epoch_boundaries")
        if not isinstance(epoch_boundaries, list):
            epoch_boundaries = list(existing.get("epoch_boundaries") or [])
            epoch_boundary = message.get("epoch_boundary")
            if isinstance(epoch_boundary, (int, float)):
                epoch_boundaries.append(epoch_boundary)
        manager.update_result(
            job_id,
            {
                "chart_data": chart_data,
                "epoch_boundaries": epoch_boundaries,
                "available_metrics": message.get(
                    "metrics", existing.get("available_metrics", [])
                ),
            },
        )


###############################################################################
def drain_worker_progress(job_id: str, worker: ProcessWorker) -> None:
    while True:
        message = worker.poll(timeout=0.0)
        if message is None:
            return
        handle_training_progress(job_id, message)


###############################################################################
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


###############################################################################
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


###############################################################################
def read_worker_result(job_id: str, worker: ProcessWorker) -> dict[str, Any]:
    result_payload = worker.read_result()
    if result_payload is None:
        if worker.exitcode not in (0, None) and not get_job_manager().should_stop(
            job_id
        ):
            raise RuntimeError(f"Training process exited with code {worker.exitcode}")
        return {}

    if "error" in result_payload and result_payload["error"]:
        failure = result_payload.get("failure")
        if isinstance(failure, dict):
            raise JobExecutionError(
                str(result_payload["error"]),
                code=str(failure.get("code", "job_failed")),
                phase=str(failure.get("phase", "execution")),
                recoverable=bool(failure.get("recoverable", True)),
            )
        raise RuntimeError(str(result_payload["error"]))

    if "result" in result_payload:
        return result_payload["result"] or {}

    return {}


###############################################################################
def register_checkpoint_result(result: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = result.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        return result
    path = Path(checkpoint_path)
    CheckpointRepository().register_completed_checkpoint(path.name, path)
    return result


###############################################################################
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


###############################################################################
def run_training_job(
    configuration: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Blocking training function that runs in background thread."""
    training_runtime = get_training_runtime()
    worker = ProcessWorker()
    training_runtime.worker = worker
    try:
        worker.start(
            target=run_training_process,
            kwargs={"configuration": configuration},
        )

        result = monitor_training_process(
            job_id,
            worker,
            stop_timeout_seconds=5.0,
        )
        return register_checkpoint_result(result)
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        worker.cleanup()
        training_runtime.worker = None


###############################################################################
def run_resume_training_job(
    checkpoint: str,
    additional_epochs: int,
    job_id: str,
) -> dict[str, Any]:
    """Blocking resume training function that runs in background thread."""
    training_runtime = get_training_runtime()
    worker = ProcessWorker()
    training_runtime.worker = worker
    try:
        worker.start(
            target=run_resume_training_process,
            kwargs={
                "checkpoint": checkpoint,
                "additional_epochs": additional_epochs,
            },
        )

        result = monitor_training_process(
            job_id,
            worker,
            stop_timeout_seconds=5.0,
        )
        return register_checkpoint_result(result)
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        worker.cleanup()
        training_runtime.worker = None


###############################################################################
class TrainingService:
    JOB_TYPE = "training"
    CHECKPOINT_EMPTY_MESSAGE = "Checkpoint name cannot be empty"
    NO_TRAINING_DATA_MESSAGE = "No training data found. Please process a dataset first."

    # -------------------------------------------------------------------------
    def __init__(
        self,
        job_manager: JobManager,
        training_runtime: TrainingRuntime,
        checkpoint_repository: CheckpointRepository,
    ) -> None:
        self.job_manager = job_manager
        self.training_runtime = training_runtime
        self.checkpoint_repository = checkpoint_repository

    # -------------------------------------------------------------------------
    def apply_runtime_training_configuration(
        self, configuration: dict[str, Any]
    ) -> None:
        server_settings = get_server_settings()
        configuration["training_seed"] = server_settings.global_settings.seed
        configuration["polling_interval"] = server_settings.jobs.polling_interval

    # -------------------------------------------------------------------------
    def initialize_job_result(
        self, job_id: str, total_epochs: int, current_epoch: int = 0
    ) -> None:
        self.job_manager.update_result(
            job_id,
            {
                "current_epoch": current_epoch,
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
            },
        )

    # -------------------------------------------------------------------------
    def build_job_start_response(
        self,
        job_id: str,
        message: str,
        initialization_error: str,
    ) -> JobStartResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise InternalServiceError(
                detail=initialization_error,
            )

        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message=message,
            poll_interval=get_server_settings().jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def get_checkpoints(self) -> CheckpointsResponse:
        """Get registered checkpoints and report explicit artifact state."""
        modser = ModelSerializer()
        checkpoints = []
        for checkpoint in self.checkpoint_repository.list_checkpoints():
            name = checkpoint.name
            try:
                if not checkpoint.artifact_complete:
                    raise ValueError("registered artifact is missing or incomplete")
                _, _, session = modser.load_training_configuration(checkpoint.path)
                epochs = session.get("epochs")
                history = session.get("history")
                loss_history = (
                    history.get("loss") if isinstance(history, dict) else None
                )
                val_loss_history = (
                    history.get("val_loss") if isinstance(history, dict) else None
                )
                if (
                    not isinstance(epochs, int)
                    or not isinstance(loss_history, list)
                    or not loss_history
                    or not isinstance(val_loss_history, list)
                    or not val_loss_history
                ):
                    raise ValueError("checkpoint session history is incomplete")
                checkpoints.append(
                    CheckpointInfo(
                        name=name,
                        epochs=epochs,
                        loss=float(loss_history[-1]),
                        val_loss=float(val_loss_history[-1]),
                        artifact_status="ready",
                    )
                )
            except Exception as exc:
                logger.warning("Failed to load checkpoint config %s: %s", name, exc)
                checkpoints.append(
                    CheckpointInfo(
                        name=name,
                        artifact_status="invalid",
                        message=str(exc),
                    )
                )

        return CheckpointsResponse(checkpoints=checkpoints)

    # -------------------------------------------------------------------------
    def get_checkpoint_metadata(self, checkpoint: str) -> CheckpointMetadataResponse:
        try:
            checkpoint = validate_checkpoint_name(checkpoint)
        except ValueError as exc:
            raise BadRequestError(
                detail=str(exc),
            ) from exc
        checkpoint_record = self.checkpoint_repository.get_checkpoint(checkpoint)
        if checkpoint_record is None:
            raise NotFoundError(
                detail=f"Checkpoint is not registered: {checkpoint}",
            )
        if not checkpoint_record.artifact_complete:
            raise InternalServiceError(
                detail=f"Checkpoint artifact is missing or incomplete: {checkpoint}",
            )

        try:
            modser = ModelSerializer()
            configuration, metadata, session = modser.load_training_configuration(
                checkpoint_record.path
            )
        except Exception as exc:
            raise InternalServiceError(
                detail=f"Failed to load checkpoint metadata: {exc}",
            ) from exc

        return CheckpointMetadataResponse(
            checkpoint=checkpoint,
            configuration=configuration,
            metadata=metadata,
            session=session,
        )

    # -------------------------------------------------------------------------
    def delete_checkpoint(self, checkpoint: str) -> DeleteResponse:
        try:
            checkpoint = validate_checkpoint_name(checkpoint)
        except ValueError as exc:
            raise BadRequestError(
                detail=str(exc),
            ) from exc

        if self.job_manager.is_job_running(self.JOB_TYPE):
            raise ConflictError(
                detail="Cannot delete checkpoints while training is active",
            )

        try:
            self.checkpoint_repository.delete_checkpoint(checkpoint)
        except CheckpointRegistryError as exc:
            if self.checkpoint_repository.get_checkpoint(checkpoint) is None:
                raise NotFoundError(detail=str(exc)) from exc
            if isinstance(exc, CheckpointReferencedError):
                raise ConflictError(detail=str(exc)) from exc
            raise InternalServiceError(detail=str(exc)) from exc

        return DeleteResponse(
            success=True,
            message=f"Deleted checkpoint {checkpoint}",
        )

    # -------------------------------------------------------------------------
    def start_training(self, request: StartTrainingRequest) -> JobStartResponse:
        if self.job_manager.is_job_running("training"):
            raise ConflictError(
                detail="Training is already in progress",
            )

        serializer = DatasetRepository()

        # Build configuration from request
        configuration = request.model_dump()

        self.apply_runtime_training_configuration(configuration)

        dataset_name = configuration.get("dataset_name")
        stored_metadata = serializer.load_training_data(
            only_metadata=True,
            dataset_name=dataset_name,
        )
        if not stored_metadata:
            raise BadRequestError(
                detail=self.NO_TRAINING_DATA_MESSAGE,
            )
        train_data, validation_data, _ = serializer.load_training_data(
            dataset_name=dataset_name
        )
        if train_data.empty and validation_data.empty:
            raise BadRequestError(
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

        self.initialize_job_result(
            job_id=job_id,
            total_epochs=configuration.get("epochs", 10),
        )

        return self.build_job_start_response(
            job_id=job_id,
            message="Training job started",
            initialization_error="Failed to initialize training job",
        )

    # -------------------------------------------------------------------------
    def resume_training(self, request: ResumeTrainingRequest) -> JobStartResponse:
        if self.job_manager.is_job_running("training"):
            raise ConflictError(
                detail="Training is already in progress",
            )

        # Initialize serializers
        serializer = DatasetRepository()
        modser = ModelSerializer()

        try:
            checkpoint = validate_checkpoint_name(request.checkpoint)
        except ValueError as exc:
            raise BadRequestError(
                detail=str(exc),
            ) from exc

        checkpoint_record = self.checkpoint_repository.get_checkpoint(checkpoint)
        if checkpoint_record is None:
            raise NotFoundError(
                detail=f"Checkpoint is not registered: {checkpoint}",
            )
        if not checkpoint_record.artifact_complete:
            raise InternalServiceError(
                detail=f"Checkpoint artifact is missing or incomplete: {checkpoint}",
            )

        try:
            train_config, _, session = modser.load_training_configuration(
                checkpoint_record.path
            )
        except Exception as exc:
            raise InternalServiceError(
                detail=f"Failed to load checkpoint metadata: {exc}",
            ) from exc

        dataset_name = str(train_config.get("dataset_name") or "").strip()
        if not dataset_name:
            raise BadRequestError(
                detail="Checkpoint configuration does not identify its processed dataset",
            )
        stored_metadata = serializer.load_training_data(
            only_metadata=True,
            dataset_name=dataset_name,
        )
        if not stored_metadata:
            raise BadRequestError(
                detail=self.NO_TRAINING_DATA_MESSAGE,
            )

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

        self.initialize_job_result(
            job_id=job_id,
            total_epochs=from_epoch + request.additional_epochs,
            current_epoch=from_epoch,
        )

        return self.build_job_start_response(
            job_id=job_id,
            message=f"Training resumed from epoch {from_epoch}",
            initialization_error="Failed to initialize training resume job",
        )


###############################################################################
@lru_cache(maxsize=1)
def get_training_service() -> TrainingService:
    return TrainingService(
        job_manager=get_job_manager(),
        training_runtime=get_training_runtime(),
        checkpoint_repository=CheckpointRepository(),
    )
