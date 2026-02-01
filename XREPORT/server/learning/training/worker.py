from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol
import multiprocessing

import os
import queue
import signal
import subprocess
import time

import pandas as pd

from XREPORT.server.utils.logger import logger
from XREPORT.server.learning.callbacks import (
    TrainingInterruptCallback,
    WorkerInterrupted,
)
from XREPORT.server.learning.device import DeviceConfig
from XREPORT.server.learning.training.dataloader import XRAYDataLoader
from XREPORT.server.learning.training.model import build_xreport_model
from XREPORT.server.learning.training.trainer import ModelTrainer
from XREPORT.server.repositories.serializer import DataSerializer, ModelSerializer


###############################################################################
class ProcessLike(Protocol):
    pid: int | None
    exitcode: int | None

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...


###############################################################################
class QueueProgressReporter:
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
    def __init__(
        self,
        progress_queue: Any,
        result_queue: Any,
        stop_event: Any,
    ) -> None:
        self.progress_queue = progress_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

    # -------------------------------------------------------------------------
    def is_interrupted(self) -> bool:
        return bool(self.stop_event.is_set())


###############################################################################
class ProcessWorker:
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

    # -------------------------------------------------------------------------
    def start(
        self,
        target: Callable[..., None],
        kwargs: dict[str, Any],
    ) -> None:
        if self.process is not None and self.process.is_alive():
            raise RuntimeError("Worker process is already running")
        self.process = self.ctx.Process(
            target=process_target,
            kwargs={
                "target": target,
                "kwargs": kwargs,
                "worker": self.as_child(),
            },
            daemon=False,
        )
        self.process.start()

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
            return message
        return None

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


###############################################################################
def process_target(
    target: Callable[..., None],
    kwargs: dict[str, Any],
    worker: WorkerChannels,
) -> None:
    if os.name != "nt":
        os.setsid()
    target(worker=worker, **kwargs)


###############################################################################
def prepare_training_data(
    configuration: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    serializer = DataSerializer()
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

    validate_paths = bool(configuration.get("validate_paths_on_train", False))
    if validate_paths:
        train_data = serializer.validate_img_paths(train_data)
        validation_data = serializer.validate_img_paths(validation_data)
        if train_data.empty and validation_data.empty:
            raise ValueError(
                "No valid images found. Image paths may have changed since dataset was processed."
            )

    return train_data, validation_data, metadata


###############################################################################
def load_resume_training_data(
    train_config: dict[str, Any],
    model_metadata: dict[str, Any],
    validate_paths: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    serializer = DataSerializer()
    current_metadata = serializer.load_training_data(only_metadata=True)
    is_validated = serializer.validate_metadata(current_metadata, model_metadata)
    if not is_validated:
        raise ValueError(
            "Current dataset metadata doesn't match checkpoint. Please reprocess the dataset."
        )

    train_data, validation_data, _ = serializer.load_training_data()
    if validate_paths:
        train_data = serializer.validate_img_paths(train_data)
        validation_data = serializer.validate_img_paths(validation_data)
        if train_data.empty or validation_data.empty:
            raise ValueError(
                "No valid images found. Image paths may have changed since dataset was processed."
            )

    return train_data, validation_data


###############################################################################
def run_training_process(
    configuration: dict[str, Any],
    worker: Any,
) -> None:
    progress_queue = worker.progress_queue
    result_queue = worker.result_queue
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
            result_queue.put({"result": {}})
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
            result_queue.put({"result": {}})
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

        result_queue.put(
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
        result_queue.put({"result": {}})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"error": str(exc)})


###############################################################################
def run_resume_training_process(
    checkpoint: str,
    additional_epochs: int,
    worker: Any,
) -> None:
    progress_queue = worker.progress_queue
    result_queue = worker.result_queue
    stop_event = worker.stop_event
    try:
        modser = ModelSerializer()
        model, train_config, model_metadata, session, checkpoint_path = (
            modser.load_checkpoint(checkpoint)
        )
        train_config["additional_epochs"] = additional_epochs

        validate_paths = bool(train_config.get("validate_paths_on_train", False))
        train_data, validation_data = load_resume_training_data(
            train_config, model_metadata, validate_paths
        )
        if train_data.empty or validation_data.empty:
            raise ValueError(
                "Training data split is empty. Reprocess the dataset or adjust "
                "sample_size/validation_size to ensure both train and validation "
                "sets contain data."
            )

        if stop_event.is_set():
            result_queue.put({"result": {}})
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

        result_queue.put(
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
        result_queue.put({"result": {}})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"error": str(exc)})
