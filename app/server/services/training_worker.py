from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol
import multiprocessing

import os
import queue
import signal
import subprocess
import time
import traceback

from server.common.utils.logger import logger
from server.repositories.serialization.dataset import (
    DatasetIntegrityError,
    DatasetRepository,
)
from server.repositories.checkpoints import CheckpointRepository

if TYPE_CHECKING:
    import pandas as pd

###############################################################################
WORKER_LIFECYCLE_MESSAGE_TYPE = "training_worker_lifecycle"
WORKER_DIAGNOSTICS_KEY = "worker_diagnostics"
MAX_WORKER_TRACEBACK_LENGTH = 12000


def _worker_exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or type(exc).__name__


def _worker_exception_traceback(exc: BaseException) -> str:
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return trace[:MAX_WORKER_TRACEBACK_LENGTH]


class ProcessLike(Protocol):

    # -------------------------------------------------------------------------
    @property
    def pid(self) -> int | None: ...

    # -------------------------------------------------------------------------
    @property
    def exitcode(self) -> int | None: ...

    # -------------------------------------------------------------------------
    def start(self) -> None: ...

    # -------------------------------------------------------------------------
    def is_alive(self) -> bool: ...

    # -------------------------------------------------------------------------
    def join(self, timeout: float | None = None) -> None: ...

###############################################################################
class QueueProgressReporter:

    # -------------------------------------------------------------------------
    def __init__(self, target_queue: Any) -> None:
        self.target_queue = target_queue

    # -------------------------------------------------------------------------
    def drain_queue(self) -> None:
        while True:
            try:
                self.target_queue.get_nowait()
            except queue.Empty:
                return
            except Exception:
                return

    # -------------------------------------------------------------------------
    def __call__(self, message: dict[str, Any]) -> None:
        try:
            if message.get("type") == "training_plot":
                self.drain_queue()
            self.target_queue.put(message, block=False)
        except queue.Full:
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to push training update: %s", exc)

###############################################################################
class WorkerChannels:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        progress_queue: Any,
        result_queue: Any,
        stop_event: Any,
    ) -> None:
        self.progress_queue = progress_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.failure_reported = False

    # -------------------------------------------------------------------------
    def is_interrupted(self) -> bool:
        return bool(self.stop_event.is_set())

    # -------------------------------------------------------------------------
    def report_lifecycle(
        self,
        phase: str,
        *,
        status: str = "started",
        **details: Any,
    ) -> None:
        message = {
            "type": WORKER_LIFECYCLE_MESSAGE_TYPE,
            "status": status,
            "phase": phase,
            "pid": os.getpid(),
            **details,
        }
        try:
            self.progress_queue.put(message, block=False)
        except queue.Full:
            return
        except (EOFError, OSError):
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to report training worker lifecycle: %s", exc)

    # -------------------------------------------------------------------------
    def report_failure(
        self,
        exc: BaseException,
        failure: dict[str, Any] | None = None,
    ) -> None:
        self.failure_reported = True
        diagnostic = {
            "status": "failed",
            "phase": "target_failed",
            "pid": os.getpid(),
            "exception_type": type(exc).__name__,
            "message": _worker_exception_message(exc),
            "traceback": _worker_exception_traceback(exc),
            "reported": True,
        }
        self.report_lifecycle(
            "target_failed",
            status="failed",
            diagnostic=diagnostic,
        )
        payload: dict[str, Any] = {
            "error": diagnostic["message"],
            WORKER_DIAGNOSTICS_KEY: diagnostic,
        }
        if failure is not None:
            payload["failure"] = failure
        try:
            self.result_queue.put(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to report training worker failure")
            raise

    # -------------------------------------------------------------------------
    def report_result(self, payload: dict[str, Any]) -> None:
        try:
            self.result_queue.put(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to report training worker result")
            raise

###############################################################################
class ProcessWorker:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        progress_queue_size: int = 256,
        result_queue_size: int = 8,
    ) -> None:
        self.ctx = multiprocessing.get_context("spawn")
        self.progress_queue = self.ctx.Queue(maxsize=progress_queue_size)
        self.result_queue = self.ctx.Queue(maxsize=result_queue_size)
        self.stop_event = self.ctx.Event()
        self.process: ProcessLike | None = None
        self.lifecycle_events: list[dict[str, Any]] = []
        self.lifecycle_phase: str | None = None

    # -------------------------------------------------------------------------
    def start(
        self,
        target: Callable[..., None],
        kwargs: dict[str, Any],
    ) -> None:
        if self.process is not None and self.process.is_alive():
            raise RuntimeError("Worker process is already running")
        process = self.ctx.Process(
            target=process_target,
            kwargs={
                "target": target,
                "kwargs": kwargs,
                "worker": self.as_child(),
            },
            daemon=False,
        )
        self.process = process
        process.start()

    # -------------------------------------------------------------------------
    def stop(self) -> None:
        self.stop_event.set()

    # -------------------------------------------------------------------------
    def interrupt(self) -> None:
        self.stop_event.set()

    # -------------------------------------------------------------------------
    def is_interrupted(self) -> bool:
        return bool(self.stop_event.is_set())

    # -------------------------------------------------------------------------
    def is_alive(self) -> bool:
        return bool(self.process is not None and self.process.is_alive())

    # -------------------------------------------------------------------------
    def join(self, timeout: float | None = None) -> None:
        if self.process is None:
            return
        self.process.join(timeout=timeout)

    # -------------------------------------------------------------------------
    def terminate(self) -> None:
        if self.process is None:
            return
        self.terminate_process_tree(self.process)

    # -------------------------------------------------------------------------
    def poll(self, timeout: float = 0.25) -> dict[str, Any] | None:
        try:
            message = self.progress_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except (EOFError, OSError):
            return None
        if isinstance(message, dict):
            self._record_lifecycle(message)
            return message
        return None

    # -------------------------------------------------------------------------
    def _record_lifecycle(self, message: dict[str, Any]) -> None:
        if message.get("type") != WORKER_LIFECYCLE_MESSAGE_TYPE:
            return
        self.lifecycle_events.append(dict(message))
        phase = message.get("phase") or message.get("status")
        if isinstance(phase, str):
            self.lifecycle_phase = phase

    # -------------------------------------------------------------------------
    def drain_progress(self) -> None:
        while True:
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                return
            except (EOFError, OSError):
                return

    # -------------------------------------------------------------------------
    def read_result(self) -> dict[str, Any] | None:
        try:
            payload = self.result_queue.get_nowait()
        except queue.Empty:
            return None
        except (EOFError, OSError):
            return None
        if isinstance(payload, dict):
            return payload
        return None

    # -------------------------------------------------------------------------
    def cleanup(self) -> None:
        self.progress_queue.close()
        self.result_queue.close()
        self.progress_queue.join_thread()
        self.result_queue.join_thread()

    # -------------------------------------------------------------------------
    def as_child(self) -> WorkerChannels:
        return WorkerChannels(
            progress_queue=self.progress_queue,
            result_queue=self.result_queue,
            stop_event=self.stop_event,
        )

    # -------------------------------------------------------------------------
    def terminate_process_tree(self, process: ProcessLike) -> None:
        pid = process.pid
        if pid is None:
            return
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", f"taskkill /PID {pid} /T /F"],
                check=False,
                capture_output=True,
            )
            return
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1)
            if process.is_alive():
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return

    # -------------------------------------------------------------------------
    @property
    def exitcode(self) -> int | None:
        if self.process is None:
            return None
        return self.process.exitcode

    # -------------------------------------------------------------------------
    @property
    def pid(self) -> int | None:
        if self.process is None:
            return None
        return self.process.pid

###############################################################################
def process_target(
    target: Callable[..., None],
    kwargs: dict[str, Any],
    worker: WorkerChannels,
) -> None:
    try:
        if os.name != "nt":
            os.setsid()
        worker.report_lifecycle("child_started", target=getattr(target, "__name__", "target"))
        worker.report_lifecycle(
            "target_started",
            target=getattr(target, "__name__", "target"),
        )
        target(worker=worker, **kwargs)
    except BaseException as exc:
        if not worker.failure_reported:
            worker.report_failure(exc)
        raise
    else:
        if worker.failure_reported:
            return
        worker.report_lifecycle(
            "worker_cancelled" if worker.is_interrupted() else "worker_completed",
            status="cancelled" if worker.is_interrupted() else "completed",
        )

###############################################################################
def prepare_training_data(
    configuration: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    serializer = DatasetRepository()
    dataset_name = configuration.get("dataset_name")
    stored_metadata = serializer.load_training_data(
        only_metadata=True,
        dataset_name=dataset_name,
    )
    if not stored_metadata:
        raise ValueError("No training metadata found. Please process a dataset first.")

    train_data, validation_data, metadata = serializer.load_training_data(
        dataset_name=dataset_name
    )
    if train_data.empty and validation_data.empty:
        raise ValueError("No training data found. Please process a dataset first.")

    if not train_data.empty:
        train_data = serializer.validate_img_paths(train_data)
    if not validation_data.empty:
        validation_data = serializer.validate_img_paths(validation_data)

    return train_data, validation_data, metadata

###############################################################################
def load_resume_training_data(
    train_config: dict[str, Any],
    model_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    serializer = DatasetRepository()
    dataset_name = str(train_config.get("dataset_name") or "").strip()
    if not dataset_name:
        raise ValueError(
            "Checkpoint configuration does not identify its processed dataset."
        )
    current_metadata = serializer.load_training_data(
        only_metadata=True,
        dataset_name=dataset_name,
    )
    is_validated = serializer.validate_metadata(current_metadata, model_metadata)
    if not is_validated:
        raise ValueError(
            "Current dataset metadata doesn't match checkpoint. Please reprocess the dataset."
        )

    train_data, validation_data, _ = serializer.load_training_data(
        dataset_name=dataset_name
    )
    if not train_data.empty:
        train_data = serializer.validate_img_paths(train_data)
    if not validation_data.empty:
        validation_data = serializer.validate_img_paths(validation_data)

    return train_data, validation_data

###############################################################################
def run_training_process(
    configuration: dict[str, Any],
    worker: Any,
) -> None:
    from server.models.callbacks import TrainingInterruptCallback, WorkerInterrupted
    from server.models.device import DeviceConfig
    from server.models.training.dataloader import XRAYDataLoader
    from server.models.training.model import build_xreport_model
    from server.models.training.trainer import ModelTrainer
    from server.repositories.serialization.model import ModelSerializer

    progress_queue = worker.progress_queue
    stop_event = worker.stop_event
    try:
        train_data, validation_data, metadata = prepare_training_data(configuration)
        if train_data.empty or validation_data.empty:
            raise ValueError(
                "Training data split is empty. Reprocess the dataset or adjust "
                "sample_size/validation_size to ensure both train and validation "
                "sets contain data."
            )

        if stop_event.is_set():
            worker.report_result({"result": {}})
            return

        logger.info("Setting device for training operations")
        device = DeviceConfig(configuration)
        device.set_device()

        modser = ModelSerializer()
        checkpoint_path = modser.create_checkpoint_folder(
            name=configuration.get("checkpoint_id")
        )

        logger.info("Building model data loaders")
        train_loader = XRAYDataLoader(
            configuration, shuffle=True
        ).build_training_dataloader(train_data)
        validation_loader = XRAYDataLoader(
            configuration, shuffle=False
        ).build_training_dataloader(validation_data)

        logger.info("Building XREPORT Transformer model")
        model = build_xreport_model(metadata, configuration)

        if stop_event.is_set():
            worker.report_result({"result": {}})
            return

        trainer = ModelTrainer(configuration)
        reporter = QueueProgressReporter(progress_queue)
        interrupt_callback = TrainingInterruptCallback(
            worker=worker,
            stop_event=stop_event,
        )

        logger.info("Starting XREPORT Transformer model training")
        trained_model, history = trainer.train_model(
            model,
            train_loader,
            validation_loader,
            checkpoint_path,
            progress_callback=reporter,
            interrupt_callback=interrupt_callback,
            worker=worker,
        )

        modser.save_pretrained_model(trained_model, checkpoint_path)
        modser.save_training_configuration(
            checkpoint_path, history, configuration, metadata
        )

        worker.report_result(
            {
                "result": {
                    "epochs": history.get("epochs", 0),
                    "final_loss": history.get("history", {}).get("loss", [0])[-1],
                    "final_val_loss": history.get("history", {}).get("val_loss", [0])[
                        -1
                    ],
                    "checkpoint_path": checkpoint_path,
                }
            }
        )
    except WorkerInterrupted:
        worker.report_result({"result": {}})
    except DatasetIntegrityError as exc:
        worker.report_failure(
            exc,
            failure={
                "code": "dataset_integrity_failed",
                "phase": "input_validation",
                "recoverable": True,
            },
        )
    except Exception as exc:  # noqa: BLE001
        worker.report_failure(exc)

###############################################################################
def run_resume_training_process(
    checkpoint: str,
    additional_epochs: int,
    worker: Any,
    poll_interval: float = 1.0,
) -> None:
    from server.models.callbacks import TrainingInterruptCallback, WorkerInterrupted
    from server.models.device import DeviceConfig
    from server.models.training.dataloader import XRAYDataLoader
    from server.models.training.trainer import ModelTrainer
    from server.repositories.serialization.model import ModelSerializer

    progress_queue = worker.progress_queue
    stop_event = worker.stop_event
    try:
        modser = ModelSerializer()
        checkpoint_record = CheckpointRepository().get_checkpoint(checkpoint)
        if checkpoint_record is None:
            raise ValueError(f"Checkpoint is not registered: {checkpoint}")
        if not checkpoint_record.artifact_complete:
            raise ValueError(
                f"Checkpoint artifact is missing or incomplete: {checkpoint}"
            )
        model, train_config, model_metadata, session, checkpoint_path = (
            modser.load_checkpoint(checkpoint_record.path)
        )
        train_config["additional_epochs"] = additional_epochs
        train_config["polling_interval"] = poll_interval

        train_data, validation_data = load_resume_training_data(
            train_config, model_metadata
        )
        if train_data.empty or validation_data.empty:
            raise ValueError(
                "Training data split is empty. Reprocess the dataset or adjust "
                "sample_size/validation_size to ensure both train and validation "
                "sets contain data."
            )

        if stop_event.is_set():
            worker.report_result({"result": {}})
            return

        logger.info("Setting device for training operations")
        device = DeviceConfig(train_config)
        device.set_device()

        logger.info("Building model data loaders")
        train_loader = XRAYDataLoader(
            train_config, shuffle=True
        ).build_training_dataloader(train_data)
        validation_loader = XRAYDataLoader(
            train_config, shuffle=False
        ).build_training_dataloader(validation_data)

        trainer = ModelTrainer(train_config, model_metadata)
        reporter = QueueProgressReporter(progress_queue)
        interrupt_callback = TrainingInterruptCallback(
            worker=worker,
            stop_event=stop_event,
        )
        from_epoch = session.get("epochs", 0)

        logger.info("Resuming training from epoch %s", from_epoch)
        trained_model, history = trainer.resume_training(
            model,
            train_loader,
            validation_loader,
            checkpoint_path,
            session=session,
            additional_epochs=additional_epochs,
            progress_callback=reporter,
            interrupt_callback=interrupt_callback,
            worker=worker,
        )

        modser.save_pretrained_model(trained_model, checkpoint_path)
        modser.save_training_configuration(
            checkpoint_path, history, train_config, model_metadata
        )

        worker.report_result(
            {
                "result": {
                    "epochs": history.get("epochs", 0),
                    "final_loss": history.get("history", {}).get("loss", [0])[-1],
                    "final_val_loss": history.get("history", {}).get("val_loss", [0])[
                        -1
                    ],
                    "checkpoint_path": checkpoint_path,
                }
            }
        )
    except WorkerInterrupted:
        worker.report_result({"result": {}})
    except DatasetIntegrityError as exc:
        worker.report_failure(
            exc,
            failure={
                "code": "dataset_integrity_failed",
                "phase": "input_validation",
                "recoverable": True,
            },
        )
    except Exception as exc:  # noqa: BLE001
        worker.report_failure(exc)
