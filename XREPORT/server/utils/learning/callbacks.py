from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from io import BytesIO
from typing import Any

import keras
import matplotlib.pyplot as plt
from keras.callbacks import Callback

from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.jobs import job_manager


###############################################################################
class WorkerInterrupted(RuntimeError):
    pass



###############################################################################
class TrainingInterruptCallback(Callback):
    def __init__(
        self,
        worker: Any | None = None,
        job_id: str | None = None,
        stop_event: Any | None = None,
    ) -> None:
        super().__init__()
        self.should_stop = False
        self.worker = worker
        self.job_id = job_id
        self.stop_event = stop_event
        self.stop_logged = False
        self.model: keras.Model

    # -------------------------------------------------------------------------
    def request_stop(self) -> None:
        self.should_stop = True
        logger.info("Training stop requested")

    # -------------------------------------------------------------------------
    def is_stop_requested(self) -> bool:
        if self.worker is not None:
            try:
                if self.worker.is_interrupted():
                    return True
            except Exception:  # noqa: BLE001
                pass
        if self.should_stop:
            return True
        if self.stop_event is not None and self.stop_event.is_set():
            return True
        if self.job_id is None:
            return False
        return job_manager.should_stop(self.job_id)

    # -------------------------------------------------------------------------
    def apply_stop_if_requested(self) -> None:
        if not self.is_stop_requested():
            return
        if not hasattr(self, "model") or self.model is None:
            return
        self.model.stop_training = True
        if not self.stop_logged:
            logger.info("Training stopped by user request")
            self.stop_logged = True

    # -------------------------------------------------------------------------
    def raise_if_interrupted(self) -> None:
        if self.worker is None:
            return
        try:
            if self.worker.is_interrupted():
                raise WorkerInterrupted("Training interrupted")
        except Exception:  # noqa: BLE001
            return

    # -------------------------------------------------------------------------
    def on_train_begin(self, logs: dict | None = None) -> None:
        self.apply_stop_if_requested()

    # -------------------------------------------------------------------------
    def on_batch_end(self, batch: int, logs: dict | None = None) -> None:
        self.raise_if_interrupted()
        self.apply_stop_if_requested()

    # -------------------------------------------------------------------------
    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        self.raise_if_interrupted()
        self.apply_stop_if_requested()

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        self.raise_if_interrupted()
        self.apply_stop_if_requested()


###############################################################################
class TrainingProgressCallback(Callback):
    def __init__(
        self,
        progress_callback: Callable[[dict[str, Any]], Any] | None,
        total_epochs: int,
        from_epoch: int = 0,
        update_interval: float = 1.0,
    ) -> None:
        super().__init__()
        self.progress_callback = progress_callback
        self.total_epochs = total_epochs
        self.from_epoch = from_epoch
        self.start_time = time.time()
        self.last_update_time = 0.0
        self.update_interval = update_interval
        self.current_epoch_index = 0
        # Store last known validation metrics (from previous epoch end)
        self.last_val_loss = 0.0
        self.last_val_accuracy = 0.0

    # -------------------------------------------------------------------------
    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self.current_epoch_index = epoch

    # -------------------------------------------------------------------------
    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        current_time = time.time()
        
        # Throttle updates based on configured interval
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        logs = logs or {}
        
        # Calculate progress based on current epoch and batch position
        steps_per_epoch = self.params.get('steps', 1)
        epoch_progress = (batch + 1) / steps_per_epoch
        total_epochs_to_run = max(1, self.total_epochs - self.from_epoch)
        completed_epochs = self.current_epoch_index - self.from_epoch
        progress_percent = int(100 * (completed_epochs + epoch_progress) / total_epochs_to_run)
        elapsed_time = current_time - self.start_time
        
        # Get training metrics from batch logs (val_* not available during batch)
        train_loss = float(logs.get("loss", 0))
        train_accuracy = float(logs.get("MaskedAccuracy", logs.get("accuracy", 0)))
        
        message = {
            "type": "training_update",
            "epoch": self.current_epoch_index + 1,
            "total_epochs": self.total_epochs,
            "progress_percent": progress_percent,
            "elapsed_seconds": int(elapsed_time),
            "loss": train_loss,
            "val_loss": self.last_val_loss,
            "accuracy": train_accuracy,
            "val_accuracy": self.last_val_accuracy,
        }

        if self.progress_callback is not None:
            self.progress_callback(message)

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        
        # Store validation metrics for next epoch's batch updates
        self.last_val_loss = float(logs.get("val_loss", 0))
        self.last_val_accuracy = float(logs.get("val_MaskedAccuracy", logs.get("val_accuracy", 0)))
        
        # Always send update at epoch end
        current_time = time.time()
        self.last_update_time = current_time
        
        processed_epochs = epoch - self.from_epoch + 1
        additional_epochs = max(1, self.total_epochs - self.from_epoch)
        progress_percent = int(100 * processed_epochs / additional_epochs)
        elapsed_time = current_time - self.start_time

        message = {
            "type": "training_update",
            "epoch": epoch + 1,
            "total_epochs": self.total_epochs,
            "progress_percent": progress_percent,
            "elapsed_seconds": int(elapsed_time),
            "loss": float(logs.get("loss", 0)),
            "val_loss": self.last_val_loss,
            "accuracy": float(logs.get("MaskedAccuracy", logs.get("accuracy", 0))),
            "val_accuracy": self.last_val_accuracy,
        }

        if self.progress_callback is not None:
            self.progress_callback(message)


###############################################################################
class RealTimeMetricsCallback(Callback):
    def __init__(
        self,
        configuration: dict[str, Any],
        checkpoint_path: str,
        past_logs: dict | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        super().__init__()
        self.configuration = configuration
        self.plot_path = os.path.join(checkpoint_path, "plots")
        os.makedirs(self.plot_path, exist_ok=True)
        
        self.total_epochs = 0 if past_logs is None else past_logs.get("epochs", 0)
        self.history: dict[str, Any] = {"history": {}, "epochs": self.total_epochs}
        self.progress_callback = progress_callback
        
        # Real-time plotting configurations
        self.update_frequency = configuration.get("update_frequency_seconds", 1.0)
        self.batch_interval = configuration.get("plot_update_batch_interval", 10)
        self.last_update_time = time.time()
        
        # Track cumulative batch index for X-axis
        self.global_batch_index = 0
        self.current_epoch_index = 0
        
        # Store batch-level data for plotting
        self.batch_history: list[dict[str, Any]] = []
        
        # Track epoch boundary batch indices for vertical separator lines
        self.epoch_boundaries: list[int] = []
        
        # Store last validation metrics for display during training batches
        self.last_val_loss = 0.0
        self.last_val_accuracy = 0.0
        
        # Track available metrics from batch logs (for chart rendering before epoch ends)
        self.available_batch_metrics: set[str] = set()

        
        # Restore past history if available
        if past_logs and "history" in past_logs:
            for metric, values in past_logs["history"].items():
                self.history["history"][metric] = list(values)
            self.history["epochs"] = past_logs.get(
                "epochs", len(next(iter(past_logs["history"].values())))
            )
            
            # Reconstruct batch history from epoch history for continuity
            # Use epoch index as batch index for resumed sessions (approximate)
            epochs_count = self.history["epochs"]
            if epochs_count > 0:
                for i in range(epochs_count):
                    point = {"batch": i + 1}
                    for metric, values in self.history["history"].items():
                        if i < len(values):
                            point[metric] = values[i]
                    self.batch_history.append(point)
                    self.epoch_boundaries.append(i + 1)
                self.global_batch_index = epochs_count

    # -------------------------------------------------------------------------
    def on_train_begin(self, logs: dict | None = None) -> None:
        """Send historical chart data immediately when training starts."""
        # If we have historical data from a resumed session, send it to the frontend
        if self.batch_history:
            self.send_plot_update(full_sync=True)

    # -------------------------------------------------------------------------
    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self.current_epoch_index = epoch

    # -------------------------------------------------------------------------
    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        return

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        
        # Store validation metrics for next epoch's batch updates
        self.last_val_loss = float(logs.get("val_loss", 0))
        self.last_val_accuracy = float(logs.get("val_MaskedAccuracy", logs.get("val_accuracy", 0)))
        
        # Update canonical history
        for key, value in logs.items():
            if key not in self.history["history"]:
                self.history["history"][key] = []
            self.history["history"][key].append(float(value))
        self.history["epochs"] = epoch + 1
        
        # Record epoch boundary at current epoch index
        self.global_batch_index = epoch + 1
        self.epoch_boundaries.append(self.global_batch_index)

        # Add epoch point to chart history
        point = {"batch": self.global_batch_index}
        for key, value in logs.items():
            point[key] = float(value)
            self.available_batch_metrics.add(key)
        self.batch_history.append(point)
        
        self.plot_training_history(save_png=False)
        self.send_plot_update(full_sync=False)

    # -------------------------------------------------------------------------
    def on_train_end(self, logs: dict | None = None) -> None:
        """Save final training history plot as PNG at end of training."""
        self.plot_training_history(save_png=True)

    # -------------------------------------------------------------------------
    def plot_training_history(self, save_png: bool = False) -> None:
        """Generate training history plot. Saves JPEG during training, PNG at end."""
        metrics = self.history["history"]
        if not metrics:
            return

        base_metrics = sorted(
            set(
                metric[4:] if metric.startswith("val_") else metric
                for metric in metrics
            )
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
        
        # Save JPEG for intermediate updates
        buffer = BytesIO()
        fig.savefig(buffer, bbox_inches="tight", format="jpeg", dpi=150)
        plot_data = buffer.getvalue()
        with open(fig_path, "wb") as target:
            target.write(plot_data)
        
        # Save final PNG at end of training
        if save_png:
            png_path = os.path.join(self.plot_path, "training_history.png")
            fig.savefig(png_path, bbox_inches="tight", format="png", dpi=200)
        
        plt.close(fig)

    # -------------------------------------------------------------------------
    def send_plot_update(self, full_sync: bool = False) -> None:
        if self.progress_callback is None:
            return

        if not self.batch_history:
            return

        # Combine metrics from epoch history and batch logs
        all_metrics = set(self.history["history"].keys()) | self.available_batch_metrics
        if full_sync:
            self.progress_callback({
                "type": "training_plot",
                "chart_data": self.batch_history,
                "metrics": list(all_metrics),
                "epochs": self.history["epochs"],
                "epoch_boundaries": self.epoch_boundaries,
            })
            return

        latest_point = self.batch_history[-1]
        payload: dict[str, Any] = {
            "type": "training_plot",
            "chart_point": latest_point,
            "metrics": list(all_metrics),
            "epochs": self.history["epochs"],
        }
        if self.epoch_boundaries:
            payload["epoch_boundary"] = self.epoch_boundaries[-1]
        self.progress_callback(payload)



###############################################################################
def initialize_training_callbacks(
    configuration: dict[str, Any],
    checkpoint_path: str,
    progress_callback: Callable[[dict[str, Any]], Any] | None = None,
    interrupt_callback: TrainingInterruptCallback | None = None,
    worker: Any | None = None,
    session: dict[str, Any] | None = None,
    total_epochs: int = 100,
) -> list[Any]:
    from_epoch = 0
    additional_epochs = configuration.get("additional_epochs", 10)
    if session:
        from_epoch = session.get("epochs", 0)
        total_epochs = additional_epochs + from_epoch

    callbacks_list: list[Any] = []

    # WebSocket progress callback
    update_frequency = configuration.get("update_frequency_seconds", 1.0)
    callbacks_list.append(
        TrainingProgressCallback(
            progress_callback, total_epochs, from_epoch, update_frequency
        )
    )

    # Training interrupt callback
    if interrupt_callback is not None:
        if worker is not None and getattr(interrupt_callback, "worker", None) is None:
            interrupt_callback.worker = worker
        callbacks_list.append(interrupt_callback)
    else:
        callbacks_list.append(TrainingInterruptCallback(worker=worker))

    # Real-time metrics plotting
    if configuration.get("plot_training_metrics", False):
        callbacks_list.append(
            RealTimeMetricsCallback(
                configuration,
                checkpoint_path,
                past_logs=session,
                progress_callback=progress_callback,
            )
        )


    # Checkpoint saving callback
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
