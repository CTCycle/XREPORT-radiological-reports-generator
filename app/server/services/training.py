from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

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
from server.services.jobs import (
    JobAlreadyRunningError,
    JobExecutionError,
    JobManager,
    get_job_manager,
)
from server.repositories.serialization.dataset import DatasetRepository
from server.repositories.checkpoints import (
    CheckpointReferencedError,
    CheckpointRegistryError,
    CheckpointRepository,
)
from server.configurations.startup import get_server_settings

if TYPE_CHECKING:
    from server.services.training_worker import ProcessWorker

###############################################################################
WORKER_EXIT_FAILURE_CODE = "training_worker_exited"
WORKER_MISSING_RESULT_CODE = "training_worker_missing_result"
WORKER_STALL_FAILURE_CODE = "training_worker_stalled"
WORKER_FAILURE_PHASE = "worker_process"
WORKER_DIAGNOSTICS_KEY = "worker_diagnostics"
DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS = 60.0
DEFAULT_WORKER_PHASE_TIMEOUT_SECONDS = 300.0
DEFAULT_WORKER_FIRST_BATCH_TIMEOUT_SECONDS = 480.0


###############################################################################
def _configured_watchdog_timeout(name: str, default: float) -> float:
    """Read an optional local watchdog override without changing persisted settings."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, raw_value)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r", name, raw_value)
        return default
    return value


###############################################################################
class TrainingRuntime:
    """Owns only the internal worker handle for the active training job."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.worker: ProcessWorker | None = None


###############################################################################
@dataclass
class _TrainingWatchdogState:
    started_at: float
    first_lifecycle_at: float | None = None
    phase: str | None = None
    phase_started_at: float | None = None
    first_batch_completed: bool = False
    last_progress: float = 0.0
    last_meaningful_progress_at: float | None = None

    # -------------------------------------------------------------------------
    def observe(self, message: dict[str, Any], now: float) -> None:
        message_type = message.get("type")
        if message_type == "training_worker_lifecycle":
            phase = message.get("phase")
            if not isinstance(phase, str) or not phase:
                return
            if self.first_lifecycle_at is None:
                self.first_lifecycle_at = now
                self.phase = phase
                self.phase_started_at = now
                self.last_meaningful_progress_at = now
            elif phase != self.phase:
                self.phase = phase
                self.phase_started_at = now
                self.last_meaningful_progress_at = now

            if phase in {"first_batch_completed", "batch_completed"}:
                self.first_batch_completed = True
                self.last_meaningful_progress_at = now
            return

        if message_type != "training_update":
            return

        progress = message.get("progress_percent")
        if not isinstance(progress, (int, float)):
            return
        numeric_progress = float(progress)
        if numeric_progress > self.last_progress:
            self.last_progress = numeric_progress
            self.last_meaningful_progress_at = now
            if numeric_progress > 0:
                self.first_batch_completed = True

    # -------------------------------------------------------------------------
    def stall_reason(
        self,
        now: float,
        *,
        startup_timeout_seconds: float,
        phase_timeout_seconds: float,
        first_batch_timeout_seconds: float,
    ) -> str | None:
        if self.first_lifecycle_at is None:
            if now - self.started_at >= startup_timeout_seconds:
                return "worker_startup_timeout"
            return None

        phase_started_at = self.phase_started_at or self.started_at
        phase_elapsed = now - phase_started_at
        if not self.first_batch_completed and self.phase in {
            "fit_entered",
            "first_batch_entered",
            "training_started",
        }:
            if phase_elapsed >= first_batch_timeout_seconds:
                return "first_batch_timeout"
            return None

        progress_reference = self.last_meaningful_progress_at or phase_started_at
        if now - progress_reference >= phase_timeout_seconds:
            return "worker_phase_timeout"
        return None

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
    if message_type == "training_worker_lifecycle":
        log_method = logger.error if message.get("status") == "failed" else logger.info
        log_method(
            "Training worker lifecycle: job_id=%s phase=%s status=%s pid=%s diagnostic=%s",
            job_id,
            message.get("phase"),
            message.get("status"),
            message.get("pid"),
            message.get("diagnostic"),
        )
        lifecycle_result: dict[str, Any] = {
            "worker_phase": str(message.get("phase") or WORKER_FAILURE_PHASE),
            "worker_phase_status": str(message.get("status") or "started"),
            "worker_phase_elapsed_seconds": message.get(
                "phase_elapsed_seconds", 0.0
            ),
            "worker_elapsed_seconds": message.get("elapsed_seconds", 0.0),
            "worker_pid": message.get("pid"),
        }
        manager.update_result(job_id, lifecycle_result)
        diagnostic = message.get("diagnostic")
        if isinstance(diagnostic, dict):
            record_worker_diagnostics(job_id, diagnostic)
    elif message_type == "training_update":
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
def drain_worker_progress(job_id: str, worker: ProcessWorker) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    while True:
        message = worker.poll(timeout=0.0)
        if message is None:
            return messages
        handle_training_progress(job_id, message)
        messages.append(message)

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
def record_worker_diagnostics(
    job_id: str,
    diagnostics: object,
) -> None:
    if not isinstance(diagnostics, dict):
        return
    public_diagnostics = {
        key: diagnostics[key]
        for key in (
            "status",
            "phase",
            "pid",
            "exitcode",
            "reported",
            "exception_type",
            "reason",
            "phase_elapsed_seconds",
            "elapsed_seconds",
        )
        if key in diagnostics
    }
    get_job_manager().update_result(
        job_id,
        {WORKER_DIAGNOSTICS_KEY: public_diagnostics},
    )


###############################################################################
def build_worker_exit_diagnostics(
    worker: ProcessWorker,
    exitcode: int,
) -> dict[str, Any]:
    return {
        "status": "exited",
        "phase": getattr(worker, "lifecycle_phase", None) or "worker_process",
        "pid": getattr(worker, "pid", None),
        "exitcode": exitcode,
        "reported": False,
    }


###############################################################################
def _read_worker_result_payload(worker: ProcessWorker) -> dict[str, Any] | None:
    return worker.read_result(timeout=0.5)


###############################################################################
def _record_reported_worker_diagnostics(
    job_id: str,
    worker: ProcessWorker,
    result_payload: dict[str, Any] | None,
) -> None:
    if result_payload is None:
        return
    diagnostics = result_payload.get(WORKER_DIAGNOSTICS_KEY)
    if isinstance(diagnostics, dict) and "phase" not in diagnostics:
        diagnostics = {
            **diagnostics,
            "phase": getattr(worker, "lifecycle_phase", None)
            or WORKER_FAILURE_PHASE,
        }
    exitcode = worker.exitcode
    if isinstance(diagnostics, dict) and exitcode not in (0, None):
        diagnostics = {**diagnostics, "exitcode": exitcode}
    if isinstance(diagnostics, dict):
        logger.error(
            "Training worker diagnostics for job %s: phase=%s pid=%s "
            "exception_type=%s message=%s traceback=%s",
            job_id,
            diagnostics.get("phase"),
            diagnostics.get("pid"),
            diagnostics.get("exception_type"),
            diagnostics.get("message"),
            diagnostics.get("traceback"),
        )
    record_worker_diagnostics(job_id, diagnostics)


###############################################################################
def _raise_missing_worker_result(
    job_id: str,
    worker: ProcessWorker,
    *,
    reported: bool,
    detail: str,
) -> NoReturn:
    diagnostics = {
        "status": "missing_result",
        "phase": getattr(worker, "lifecycle_phase", None) or WORKER_FAILURE_PHASE,
        "pid": getattr(worker, "pid", None),
        "exitcode": worker.exitcode,
        "reported": reported,
    }
    record_worker_diagnostics(job_id, diagnostics)
    raise JobExecutionError(
        detail,
        code=WORKER_MISSING_RESULT_CODE,
        phase=str(diagnostics["phase"]),
        recoverable=True,
    )


###############################################################################
def read_worker_result(job_id: str, worker: ProcessWorker) -> dict[str, Any]:
    manager = get_job_manager()
    result_payload = _read_worker_result_payload(worker)
    stop_requested = manager.should_stop(job_id)
    exitcode = worker.exitcode
    _record_reported_worker_diagnostics(job_id, worker, result_payload)

    if exitcode not in (0, None) and not stop_requested:
        if not (result_payload and result_payload.get("error")):
            diagnostics = build_worker_exit_diagnostics(worker, exitcode)
            record_worker_diagnostics(job_id, diagnostics)
            raise JobExecutionError(
                f"Training process exited with code {exitcode}",
                code=WORKER_EXIT_FAILURE_CODE,
                phase=WORKER_FAILURE_PHASE,
                recoverable=True,
            )

    if result_payload is None:
        if stop_requested:
            return {}
        _raise_missing_worker_result(
            job_id,
            worker,
            reported=False,
            detail="Training worker exited without a result payload",
        )

    if result_payload.get("error"):
        failure = result_payload.get("failure")
        if isinstance(failure, dict):
            raise JobExecutionError(
                str(result_payload["error"]),
                code=str(failure.get("code", "job_failed")),
                phase=str(failure.get("phase", "execution")),
                recoverable=bool(failure.get("recoverable", True)),
            )
        raise JobExecutionError(str(result_payload["error"]))

    result = result_payload.get("result")
    if isinstance(result, dict) and result:
        return result
    if stop_requested:
        return {}
    _raise_missing_worker_result(
        job_id,
        worker,
        reported=True,
        detail=(
            "Training worker returned an empty result payload"
            if "result" in result_payload
            else "Training worker returned no success or failure payload"
        ),
    )

###############################################################################
def register_checkpoint_result(result: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = result.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        return result
    path = Path(checkpoint_path)
    CheckpointRepository().register_completed_checkpoint(path.name, path)
    return result


###############################################################################
def _training_stall_diagnostics(
    worker: ProcessWorker,
    watchdog: _TrainingWatchdogState,
    reason: str,
    now: float,
) -> dict[str, Any]:
    phase = watchdog.phase or getattr(worker, "lifecycle_phase", None)
    phase = phase or WORKER_FAILURE_PHASE
    phase_started_at = watchdog.phase_started_at or watchdog.started_at
    return {
        "status": "stalled",
        "phase": phase,
        "pid": getattr(worker, "pid", None),
        "exitcode": getattr(worker, "exitcode", None),
        "reported": False,
        "reason": reason,
        "phase_elapsed_seconds": round(max(0.0, now - phase_started_at), 3),
        "elapsed_seconds": round(max(0.0, now - watchdog.started_at), 3),
    }


###############################################################################
def _record_training_stall(
    job_id: str,
    worker: ProcessWorker,
    watchdog: _TrainingWatchdogState,
    reason: str,
    now: float,
) -> dict[str, Any]:
    diagnostics = _training_stall_diagnostics(worker, watchdog, reason, now)
    record_worker_diagnostics(job_id, diagnostics)
    get_job_manager().update_result(
        job_id,
        {
            "worker_phase": diagnostics["phase"],
            "worker_phase_status": "stalled",
            "worker_phase_elapsed_seconds": diagnostics["phase_elapsed_seconds"],
            "worker_elapsed_seconds": diagnostics["elapsed_seconds"],
        },
    )
    logger.warning(
        "Training worker stalled: job_id=%s phase=%s reason=%s phase_elapsed=%.3fs",
        job_id,
        diagnostics["phase"],
        reason,
        diagnostics["phase_elapsed_seconds"],
    )
    return diagnostics

###############################################################################
def monitor_training_process(  # noqa: C901 - watchdog and cancellation states are intentionally explicit
    job_id: str,
    worker: ProcessWorker,
    stop_timeout_seconds: float,
    *,
    startup_timeout_seconds: float | None = None,
    phase_timeout_seconds: float | None = None,
    first_batch_timeout_seconds: float | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    clock_fn = clock or time.monotonic
    started_at = getattr(worker, "started_at", None)
    if not isinstance(started_at, (int, float)):
        started_at = clock_fn()
    watchdog = _TrainingWatchdogState(started_at=float(started_at))
    startup_timeout = (
        startup_timeout_seconds
        if startup_timeout_seconds is not None
        else _configured_watchdog_timeout(
            "XREPORT_TRAINING_STARTUP_TIMEOUT_SECONDS",
            DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS,
        )
    )
    phase_timeout = (
        phase_timeout_seconds
        if phase_timeout_seconds is not None
        else _configured_watchdog_timeout(
            "XREPORT_TRAINING_PHASE_TIMEOUT_SECONDS",
            DEFAULT_WORKER_PHASE_TIMEOUT_SECONDS,
        )
    )
    first_batch_timeout = (
        first_batch_timeout_seconds
        if first_batch_timeout_seconds is not None
        else _configured_watchdog_timeout(
            "XREPORT_TRAINING_FIRST_BATCH_TIMEOUT_SECONDS",
            DEFAULT_WORKER_FIRST_BATCH_TIMEOUT_SECONDS,
        )
    )
    stop_requested_at: float | None = None
    stall_reason: str | None = None
    stall_diagnostics: dict[str, Any] | None = None

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

        now = clock_fn()
        if stall_reason is None and not get_job_manager().should_stop(job_id):
            stall_reason = watchdog.stall_reason(
                now,
                startup_timeout_seconds=startup_timeout,
                phase_timeout_seconds=phase_timeout,
                first_batch_timeout_seconds=first_batch_timeout,
            )
            if stall_reason is not None:
                stall_diagnostics = _record_training_stall(
                    job_id,
                    worker,
                    watchdog,
                    stall_reason,
                    now,
                )
                worker.stop()
                stop_requested_at = now

        if stall_reason is not None:
            if now - (stop_requested_at or now) >= stop_timeout_seconds:
                worker.terminate()
                break
            continue

        message = worker.poll(timeout=0.25)
        if message is not None:
            handle_training_progress(job_id, message)
            watchdog.observe(message, clock_fn())
            for queued_message in drain_worker_progress(job_id, worker):
                watchdog.observe(queued_message, clock_fn())

    worker.join(timeout=5)
    drain_worker_progress(job_id, worker)

    if stall_reason is not None:
        phase = (stall_diagnostics or {}).get("phase", WORKER_FAILURE_PHASE)
        raise JobExecutionError(
            f"Training worker stalled during phase {phase}",
            code=WORKER_STALL_FAILURE_CODE,
            phase=str(phase),
            recoverable=True,
        )

    return read_worker_result(job_id=job_id, worker=worker)

###############################################################################
def run_training_job(
    configuration: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Blocking training function that runs in background thread."""
    from server.services.training_worker import (
        ProcessWorker,
        run_training_process,
    )

    training_runtime = get_training_runtime()
    worker = ProcessWorker()
    training_runtime.worker = worker
    try:
        worker.start(
            target=run_training_process,
            kwargs={"configuration": configuration},
            job_id=job_id,
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
    poll_interval: float = 1.0,
) -> dict[str, Any]:
    """Blocking resume training function that runs in background thread."""
    from server.services.training_worker import (
        ProcessWorker,
        run_resume_training_process,
    )

    training_runtime = get_training_runtime()
    worker = ProcessWorker()
    training_runtime.worker = worker
    try:
        worker.start(
            target=run_resume_training_process,
            kwargs={
                "checkpoint": checkpoint,
                "additional_epochs": additional_epochs,
                "poll_interval": poll_interval,
            },
            job_id=job_id,
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
            self.build_initial_job_result(
                total_epochs=total_epochs,
                current_epoch=current_epoch,
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def build_initial_job_result(
        total_epochs: int,
        current_epoch: int = 0,
    ) -> dict[str, Any]:
        """Build the result snapshot before a worker thread can publish updates."""

        return {
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "loss": 0.0,
            "val_loss": 0.0,
            "accuracy": 0.0,
            "val_accuracy": 0.0,
            "progress_percent": 0,
            "elapsed_seconds": 0,
            "worker_phase": "starting",
            "worker_phase_status": "pending",
            "worker_phase_elapsed_seconds": 0.0,
            "worker_elapsed_seconds": 0.0,
            "worker_pid": None,
            "chart_data": [],
            "epoch_boundaries": [],
            "available_metrics": [],
        }

    # -------------------------------------------------------------------------
    def build_job_start_response(
        self,
        job_id: str,
        message: str,
        initialization_error: str,
        poll_interval: float | None = None,
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
            poll_interval=(
                get_server_settings().jobs.polling_interval
                if poll_interval is None
                else poll_interval
            ),
        )

    # -------------------------------------------------------------------------
    def get_checkpoints(self) -> CheckpointsResponse:
        """Get registered checkpoints and report explicit artifact state."""
        from server.repositories.serialization.model import ModelSerializer

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
            from server.repositories.serialization.model import ModelSerializer

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
        poll_interval = float(configuration["polling_interval"])

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
        try:
            job_id = self.job_manager.start_job(
                job_type="training",
                runner=run_training_job,
                poll_interval=poll_interval,
                initial_result=self.build_initial_job_result(
                    total_epochs=configuration.get("epochs", 10),
                ),
                kwargs={
                    "configuration": configuration,
                },
                require_idle=True,
            )
        except JobAlreadyRunningError as exc:
            raise ConflictError(detail="Training is already in progress") from exc

        return self.build_job_start_response(
            job_id=job_id,
            message="Training job started",
            initialization_error="Failed to initialize training job",
            poll_interval=poll_interval,
        )

    # -------------------------------------------------------------------------
    def resume_training(self, request: ResumeTrainingRequest) -> JobStartResponse:
        if self.job_manager.is_job_running("training"):
            raise ConflictError(
                detail="Training is already in progress",
            )

        # Initialize serializers
        serializer = DatasetRepository()
        from server.repositories.serialization.model import ModelSerializer

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
        poll_interval = get_server_settings().jobs.polling_interval

        # Start background job
        try:
            job_id = self.job_manager.start_job(
                job_type="training",
                runner=run_resume_training_job,
                poll_interval=poll_interval,
                initial_result=self.build_initial_job_result(
                    total_epochs=from_epoch + request.additional_epochs,
                    current_epoch=from_epoch,
                ),
                kwargs={
                    "checkpoint": checkpoint,
                    "additional_epochs": request.additional_epochs,
                    "poll_interval": poll_interval,
                },
                require_idle=True,
            )
        except JobAlreadyRunningError as exc:
            raise ConflictError(detail="Training is already in progress") from exc

        return self.build_job_start_response(
            job_id=job_id,
            message=f"Training resumed from epoch {from_epoch}",
            initialization_error="Failed to initialize training resume job",
            poll_interval=poll_interval,
        )

###############################################################################
@lru_cache(maxsize=1)
def get_training_service() -> TrainingService:
    return TrainingService(
        job_manager=get_job_manager(),
        training_runtime=get_training_runtime(),
        checkpoint_repository=CheckpointRepository(),
    )
