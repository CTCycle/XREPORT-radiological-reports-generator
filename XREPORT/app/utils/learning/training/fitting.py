from __future__ import annotations

from typing import Any

from keras import Model
from keras.utils import set_random_seed

from XREPORT.app.utils.learning.callbacks import initialize_callbacks_handler


# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
###############################################################################
class ModelTraining:
    def __init__(
        self, configuration: dict[str, Any], metadata: Any | None = None
    ) -> None:
        set_random_seed(configuration.get("training_seed", 42))
        self.configuration = configuration
        self.metadata = metadata

    # -------------------------------------------------------------------------
    def train_model(
        self,
        model: Model,
        train_data: Any,
        validation_data: Any,
        checkpoint_path: str,
        **kwargs,
    ) -> tuple[Model, dict[str, Any]]:
        """
        Train the supplied model from scratch using the configured callbacks.

        Keyword arguments:
            model: Keras model configured for caption generation.
            train_data: Dataset or generator yielding training batches.
            validation_data: Dataset providing validation batches.
            checkpoint_path: Directory used to persist checkpoints and logs.
            **kwargs: Optional hooks for progress callbacks and worker threads.

        Return value:
            Tuple containing the trained model and a dictionary summarizing the
            training history.
        """
        total_epochs = self.configuration.get("epochs", 10)
        # add all callbacks to the callback list
        callbacks_list = initialize_callbacks_handler(
            self.configuration,
            checkpoint_path,
            total_epochs=total_epochs,
            progress_callback=kwargs.get("progress_callback", None),
            worker=kwargs.get("worker", None),
        )

        # run model fit using keras API method.
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
        session: dict[str, Any] = {},
        additional_epochs: int = 10,
        **kwargs,
    ) -> tuple[Model, dict[str, Any]]:
        """
        Resume a partially trained model and extend training for more epochs.

        Keyword arguments:
            model: Keras model instance restored from previous checkpoints.
            train_data: Dataset or generator yielding training batches.
            validation_data: Dataset providing validation batches.
            checkpoint_path: Directory used to persist checkpoints and logs.
            session: Serialized history returned by a previous training run.
            additional_epochs: Extra epochs to execute beyond the stored state.
            **kwargs: Optional hooks for progress callbacks and worker threads.

        Return value:
            Tuple containing the updated model and the merged training history.
        """
        from_epoch = 0 if not session else session["epochs"]
        total_epochs = from_epoch + additional_epochs
        # add all callbacks to the callback list
        callbacks_list = initialize_callbacks_handler(
            self.configuration,
            checkpoint_path,
            session,
            total_epochs,
            progress_callback=kwargs.get("progress_callback", None),
            worker=kwargs.get("worker", None),
        )

        # run model fit using keras API method.
        new_session = model.fit(
            train_data,
            epochs=total_epochs,
            validation_data=validation_data,
            callbacks=callbacks_list,
            initial_epoch=from_epoch,
        )

        # update history with new scores and final epoch value
        session_keys = session["history"].keys()
        new_history = {
            k: session["history"][k] + new_session.history[k] for k in session_keys
        }
        history = {"history": new_history, "epochs": new_session.epoch[-1] + 1}

        return model, history
