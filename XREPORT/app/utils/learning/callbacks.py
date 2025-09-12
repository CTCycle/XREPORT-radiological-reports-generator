from __future__ import annotations

import os
import subprocess
import time
import webbrowser
from typing import Any

import keras
import matplotlib.pyplot as plt
from keras.callbacks import Callback

from XREPORT.app.client.workers import (
    ProcessWorker,
    ThreadWorker,
    WorkerInterrupted,
)
from XREPORT.app.logger import logger


# [CALLBACK FOR UI PROGRESS BAR]
###############################################################################
class ProgressBarCallback(Callback):
    def __init__(
        self,
        progress_callback: Any | None,
        total_epochs: int,
        from_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.progress_callback = progress_callback
        self.total_epochs = total_epochs
        self.from_epoch = from_epoch

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        processed_epochs = epoch - self.from_epoch + 1
        additional_epochs = max(1, self.total_epochs - self.from_epoch)
        percent = int(100 * processed_epochs / additional_epochs)
        if self.progress_callback is not None:
            self.progress_callback(percent)


# [CALLBACK FOR TRAIN INTERRUPTION]
###############################################################################
class LearningInterruptCallback(Callback):
    def __init__(self, worker: ThreadWorker | ProcessWorker | None = None) -> None:
        super().__init__()
        self.worker = worker

    # -------------------------------------------------------------------------
    def on_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        if self.worker is not None and self.worker.is_interrupted():
            self.model.stop_training = True
            raise WorkerInterrupted()

    # -------------------------------------------------------------------------
    def on_validation_batch_end(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        if self.worker is not None and self.worker.is_interrupted():
            raise WorkerInterrupted()


# [CALLBACK FOR REAL TIME TRAINING MONITORING]
###############################################################################
class RealTimeHistory(Callback):
    def __init__(
        self, plot_path: str, past_logs: dict[str, Any] | None = None, **kwargs
    ) -> None:
        super(RealTimeHistory, self).__init__(**kwargs)
        self.plot_path = plot_path
        os.makedirs(self.plot_path, exist_ok=True)
        # Separate dicts for training vs. validation metrics
        self.total_epochs: int = (
            0 if past_logs is None else int(past_logs.get("epochs", 0))
        )
        self.history: dict[str, Any] = {"history": {}, "epochs": self.total_epochs}

        # If past_logs provided, split into history and val_history
        if past_logs and "history" in past_logs:
            for metric, values in past_logs["history"].items():
                self.history["history"][metric] = list(values)
            self.history["epochs"] = past_logs.get(
                "epochs", len(next(iter(past_logs["history"].values())))
            )

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        for key, value in logs.items():
            if key not in self.history["history"]:
                self.history["history"][key] = []
            self.history["history"][key].append(value)
        self.history["epochs"] = epoch + 1
        self.plot_training_history()

    # -------------------------------------------------------------------------
    def plot_training_history(self) -> None:
        fig_path = os.path.join(self.plot_path, "training_history.jpeg")
        plt.figure(figsize=(16, 14))
        metrics = self.history["history"]
        # Find unique base metric names
        base_metrics = sorted(
            set(m[4:] if m.startswith("val_") else m for m in metrics.keys())
        )

        plt.figure(figsize=(16, 5 * len(base_metrics)))
        for i, base in enumerate(base_metrics):
            plt.subplot(len(base_metrics), 1, i + 1)
            # Plot training metric
            if base in metrics:
                plt.plot(metrics[base], label="train")
            # Plot validation metric if exists
            val_key = f"val_{base}"
            if val_key in metrics:
                plt.plot(metrics[val_key], label="val")
            plt.title(base)
            plt.ylabel("")
            plt.xlabel("Epoch")
            plt.legend(loc="best", fontsize=10)

        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight", format="jpeg", dpi=300)
        plt.close()


# [CALLBACKS HANDLER]
###############################################################################
def initialize_callbacks_handler(
    configuration: dict[str, Any],
    checkpoint_path: str,
    session: dict[str, Any] | None = None,
    total_epochs: int = 100,
    **kwargs,
) -> list[Callback]:
    from_epoch = 0
    additional_epochs = configuration.get("additional_epochs", 10)
    if session:
        from_epoch = int(session["epochs"])  # type: ignore[index]
        total_epochs = additional_epochs + from_epoch

    callbacks_list = [
        ProgressBarCallback(
            kwargs.get("progress_callback", None), total_epochs, from_epoch
        ),
        LearningInterruptCallback(kwargs.get("worker", None)),
    ]

    if configuration.get("plot_training_metrics", False):
        callbacks_list.append(RealTimeHistory(checkpoint_path, past_logs=session))

    if configuration.get("use_tensorboard", False):
        logger.debug("Using tensorboard during training")
        log_path = os.path.join(checkpoint_path, "tensorboard")
        callbacks_list.append(
            keras.callbacks.TensorBoard(
                log_dir=log_path, histogram_freq=1, write_images=True
            )
        )
        start_tensorboard_subprocess(log_path)

    if configuration.get("save_checkpoints", False):
        logger.debug("Adding checkpoint saving callback")
        checkpoint_filepath = os.path.join(
            checkpoint_path, "model_checkpoint_E{epoch:02d}.keras"
        )
        callbacks_list.append(
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor="val_loss",
                save_best_only=False,
                mode="auto",
                verbose=0,
            )
        )

    return callbacks_list


###############################################################################
def start_tensorboard_subprocess(log_dir: str):
    tensorboard_command = ["tensorboard", "--logdir", log_dir, "--port", "6006"]
    subprocess.Popen(
        tensorboard_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(4)
    webbrowser.open("http://localhost:6006")
