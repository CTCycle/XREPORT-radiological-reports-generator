from __future__ import annotations

from collections.abc import Callable
from typing import Any

from keras import Model
from keras.utils import set_random_seed

from server.common.utils.logger import logger
from server.models.callbacks import (
    TrainingInterruptCallback,
    initialize_training_callbacks,
)

###############################################################################
class ModelTrainer:
    """Run Keras ``fit`` on torch DataLoaders.

    Callers build and fit inside ``keras.device(...)`` (see ``DeviceConfig``);
    Keras moves each batch to that device. ``verbose=0`` because progress is
    reported by ``TrainingProgressCallback``; the progress bar would force a
    device sync and a stdout write on every batch.
    """

    # -------------------------------------------------------------------------
    def __init__(
        self, configuration: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> None:
        training_seed = configuration.get("training_seed", 42)
        set_random_seed(training_seed)
        self.configuration = configuration
        self.metadata = metadata

    # -------------------------------------------------------------------------
    def train_model(
        self,
        model: Model,
        train_data: Any,
        validation_data: Any,
        checkpoint_path: str,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
        interrupt_callback: TrainingInterruptCallback | None = None,
        worker: Any | None = None,
    ) -> tuple[Model, dict[str, Any]]:
        total_epochs = self.configuration.get("epochs", 10)

        callbacks_list = initialize_training_callbacks(
            self.configuration,
            checkpoint_path,
            progress_callback=progress_callback,
            interrupt_callback=interrupt_callback,
            worker=worker,
            total_epochs=total_epochs,
            metadata=self.metadata,
        )

        logger.info(f"Starting training for {total_epochs} epochs")

        session = model.fit(
            train_data,
            epochs=total_epochs,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=0,
        )

        history = {"history": session.history, "epochs": session.epoch[-1] + 1}

        return model, history

    # -------------------------------------------------------------------------
    def resume_training(
        self,
        model: Model,
        train_data: Any,
        validation_data: Any,
        checkpoint_path: str,
        session: dict[str, Any] | None = None,
        additional_epochs: int = 10,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
        interrupt_callback: TrainingInterruptCallback | None = None,
        worker: Any | None = None,
    ) -> tuple[Model, dict[str, Any]]:
        session = session or {}
        from_epoch = session.get("epochs", 0)
        total_epochs = from_epoch + additional_epochs

        callbacks_list = initialize_training_callbacks(
            self.configuration,
            checkpoint_path,
            progress_callback=progress_callback,
            interrupt_callback=interrupt_callback,
            worker=worker,
            session=session,
            total_epochs=total_epochs,
            metadata=self.metadata,
        )

        logger.info(f"Resuming training from epoch {from_epoch} to {total_epochs}")

        new_session = model.fit(
            train_data,
            epochs=total_epochs,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=0,
            initial_epoch=from_epoch,
        )

        if session and "history" in session:
            session_keys = session["history"].keys()
            new_history = {
                k: session["history"][k] + new_session.history[k] for k in session_keys
            }
        else:
            new_history = new_session.history

        history = {"history": new_history, "epochs": new_session.epoch[-1] + 1}

        return model, history
