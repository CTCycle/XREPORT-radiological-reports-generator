from __future__ import annotations

from collections.abc import Callable
from typing import Any

from keras import Model
from keras.utils import set_random_seed

from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.learning.callbacks import (
    TrainingInterruptCallback,
    initialize_training_callbacks,
)


###############################################################################
class ModelTrainer:
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
        websocket_callback: Callable[[dict[str, Any]], Any] | None = None,
        interrupt_callback: TrainingInterruptCallback | None = None,
    ) -> tuple[Model, dict[str, Any]]:
        total_epochs = self.configuration.get("epochs", 10)
        
        callbacks_list = initialize_training_callbacks(
            self.configuration,
            checkpoint_path,
            websocket_callback=websocket_callback,
            interrupt_callback=interrupt_callback,
            total_epochs=total_epochs,
        )

        logger.info(f"Starting training for {total_epochs} epochs")
        
        # Run model fit
        session = model.fit(
            train_data,
            epochs=total_epochs,
            validation_data=validation_data,
            callbacks=callbacks_list,
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
        websocket_callback: Callable[[dict[str, Any]], Any] | None = None,
        interrupt_callback: TrainingInterruptCallback | None = None,
    ) -> tuple[Model, dict[str, Any]]:
        session = session or {}
        from_epoch = session.get("epochs", 0)
        total_epochs = from_epoch + additional_epochs
        
        callbacks_list = initialize_training_callbacks(
            self.configuration,
            checkpoint_path,
            websocket_callback=websocket_callback,
            interrupt_callback=interrupt_callback,
            session=session,
            total_epochs=total_epochs,
        )

        logger.info(f"Resuming training from epoch {from_epoch} to {total_epochs}")
        
        # Run model fit
        new_session = model.fit(
            train_data,
            epochs=total_epochs,
            validation_data=validation_data,
            callbacks=callbacks_list,
            initial_epoch=from_epoch,
        )

        # Update history with new scores
        if session and "history" in session:
            session_keys = session["history"].keys()
            new_history = {
                k: session["history"][k] + new_session.history[k] for k in session_keys
            }
        else:
            new_history = new_session.history

        history = {"history": new_history, "epochs": new_session.epoch[-1] + 1}

        return model, history
