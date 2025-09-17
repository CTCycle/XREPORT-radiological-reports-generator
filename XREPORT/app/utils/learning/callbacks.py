from __future__ import annotations

from collections.abc import Callable
import os
import subprocess
import time
import webbrowser
from io import BytesIO
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
        progress_callback: Callable[[int], Any] | None,
        total_epochs: int,
        from_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.progress_callback = progress_callback
        self.total_epochs = total_epochs
        self.from_epoch = from_epoch

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs: dict | None = None) -> None:
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
        self.model: keras.Model

    # -------------------------------------------------------------------------
    def on_batch_end(self, batch, logs: dict | None = None) -> None:
        if self.worker is not None and self.worker.is_interrupted():
            self.model.stop_training = True
            raise WorkerInterrupted()

    # -------------------------------------------------------------------------
    def on_validation_batch_end(self, batch, logs: dict | None = None) -> None:
        if self.worker is not None and self.worker.is_interrupted():
            raise WorkerInterrupted()


# [CALLBACK FOR REAL TIME TRAINING MONITORING]
###############################################################################
class RealTimeHistory(Callback):
    def __init__(
        self,
        plot_path: str,
        past_logs: dict | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.plot_path = plot_path
        os.makedirs(self.plot_path, exist_ok=True)
        self.total_epochs = 0 if past_logs is None else past_logs.get("epochs", 0)
        self.history = {"history": {}, "epochs": self.total_epochs}
        self.progress_callback = progress_callback

        if past_logs and "history" in past_logs:
            for metric, values in past_logs["history"].items():
                self.history["history"][metric] = list(values)
            self.history["epochs"] = past_logs.get(
                "epochs", len(next(iter(past_logs["history"].values())))
            )

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs: dict | None = None) -> None:
        logs = logs or {}
        for key, value in logs.items():
            if key not in self.history["history"]:
                self.history["history"][key] = []
            self.history["history"][key].append(value)
        self.history["epochs"] = epoch + 1
        self.plot_training_history()

    # -------------------------------------------------------------------------
    def plot_training_history(self) -> None:
        metrics = self.history["history"]
        if not metrics:
            return

        base_metrics = sorted(
            set(metric[4:] if metric.startswith("val_") else metric for metric in metrics)
        )
        if not base_metrics:
            return

        fig_path = os.path.join(self.plot_path, "training_history.jpeg")
        rows = len(base_metrics)
        fig, axes = plt.subplots(rows, 1, figsize=(16, 5 * rows))
        if rows == 1:
            axes = [axes]

        for ax, base in zip(axes, base_metrics):
            if base in metrics:
                ax.plot(metrics[base], label="train")
            val_key = f"val_{base}"
            if val_key in metrics:
                ax.plot(metrics[val_key], label="val")
            ax.set_title(base)
            ax.set_ylabel("")
            ax.set_xlabel("Epoch")
            ax.legend(loc="best", fontsize=10)

        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, bbox_inches="tight", format="jpeg", dpi=300)
        data = buffer.getvalue()
        with open(fig_path, "wb") as target:
            target.write(data)
        if self.progress_callback:
            self.progress_callback(
                {"kind": "render", "source": "train_metrics", "stream": "history", "data": data}
            )
        plt.close(fig)

# [CALLBACKS HANDLER]
###############################################################################
# [CALLBACKS HANDLER]
###############################################################################
def initialize_callbacks_handler(
    configuration: dict[str, Any],
    checkpoint_path: str,
    session: dict = {},
    total_epochs: int = 100,
    **kwargs,
) -> list[Any]:
    from_epoch = 0
    additional_epochs = configuration.get("additional_epochs", 10)
    if session:
        from_epoch = session["epochs"]
        total_epochs = additional_epochs + from_epoch

    callbacks_list = [
        ProgressBarCallback(
            kwargs.get("progress_callback", None), total_epochs, from_epoch
        ),
        LearningInterruptCallback(kwargs.get("worker", None)),
    ]

    if configuration.get("plot_training_metrics", False):
        callbacks_list.append(
            RealTimeHistory(
                checkpoint_path,
                past_logs=session,
                progress_callback=kwargs.get("progress_callback", None),
            )
        )

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
def start_tensorboard_subprocess(log_dir: str) -> None:
    tensorboard_command = ["tensorboard", "--logdir", log_dir, "--port", "6006"]
    subprocess.Popen(
        tensorboard_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(4)
    webbrowser.open("http://localhost:6006")
