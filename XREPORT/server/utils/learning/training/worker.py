from __future__ import annotations

from typing import Any
import queue

import pandas as pd

from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.learning.callbacks import TrainingInterruptCallback
from XREPORT.server.utils.learning.device import DeviceConfig
from XREPORT.server.utils.learning.processing import TrainValidationSplit
from XREPORT.server.utils.learning.training.dataloader import XRAYDataLoader
from XREPORT.server.utils.learning.training.model import build_xreport_model
from XREPORT.server.utils.learning.training.trainer import ModelTrainer
from XREPORT.server.utils.repository.serializer import DataSerializer, ModelSerializer


###############################################################################
class QueueProgressReporter:
    def __init__(self, target_queue: Any) -> None:
        self.target_queue = target_queue

    # -------------------------------------------------------------------------
    def __call__(self, message: dict[str, Any]) -> None:
        try:
            self.target_queue.put(message, block=False)
        except queue.Full:
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to push training update: %s", exc)


###############################################################################
def prepare_training_data(
    configuration: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    serializer = DataSerializer()
    stored_metadata = serializer.load_training_data(only_metadata=True)
    if not stored_metadata:
        raise ValueError("No training metadata found. Please process a dataset first.")

    train_data, validation_data, metadata = serializer.load_training_data()
    if train_data.empty and validation_data.empty:
        raise ValueError("No training data found. Please process a dataset first.")

    req_sample = float(configuration.get("sample_size", 1.0))
    req_val = float(configuration.get("validation_size", 0.2))
    stored_sample = float(stored_metadata.get("sample_size", 1.0))
    stored_val = float(stored_metadata.get("validation_size", 0.2))

    if abs(req_sample - stored_sample) > 1e-6 or abs(req_val - stored_val) > 1e-6:
        logger.info("Splitting parameters changed, re-splitting data")
        full_data = pd.concat([train_data, validation_data], ignore_index=True)

        if abs(req_sample - stored_sample) > 1e-6:
            if req_sample < stored_sample:
                frac = req_sample / stored_sample
                full_data = full_data.sample(
                    frac=frac,
                    random_state=int(configuration.get("training_seed", 42)),
                )
            else:
                logger.warning(
                    "Requested sample_size is larger than stored sample_size. "
                    "Using all available data."
                )

        splitter = TrainValidationSplit(
            {
                "validation_size": req_val,
                "split_seed": int(configuration.get("training_seed", 42)),
            },
            full_data,
        )
        split_data = splitter.split_train_and_validation()

        train_data = split_data[split_data["split"] == "train"].reset_index(drop=True)
        validation_data = split_data[split_data["split"] == "validation"].reset_index(
            drop=True
        )
        metadata["sample_size"] = req_sample
        metadata["validation_size"] = req_val
        metadata["seed"] = int(configuration.get("training_seed", 42))

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    serializer = DataSerializer()
    current_metadata = serializer.load_training_data(only_metadata=True)
    is_validated = serializer.validate_metadata(current_metadata, model_metadata)
    if not is_validated:
        raise ValueError(
            "Current dataset metadata doesn't match checkpoint. Please reprocess the dataset."
        )

    train_data, validation_data, _ = serializer.load_training_data()
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
    progress_queue: Any,
    result_queue: Any,
    stop_event: Any,
) -> None:
    try:
        train_data, validation_data, metadata = prepare_training_data(configuration)

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
        train_loader = XRAYDataLoader(configuration, shuffle=True).build_training_dataloader(
            train_data
        )
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
        interrupt_callback = TrainingInterruptCallback(stop_event=stop_event)

        logger.info("Starting XREPORT Transformer model training")
        trained_model, history = trainer.train_model(
            model,
            train_loader,
            validation_loader,
            checkpoint_path,
            websocket_callback=reporter,
            interrupt_callback=interrupt_callback,
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
                    "final_val_loss": history.get("history", {}).get("val_loss", [0])[-1],
                    "checkpoint_path": checkpoint_path,
                }
            }
        )
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"error": str(exc)})


###############################################################################
def run_resume_training_process(
    checkpoint: str,
    additional_epochs: int,
    progress_queue: Any,
    result_queue: Any,
    stop_event: Any,
) -> None:
    try:
        modser = ModelSerializer()
        model, train_config, model_metadata, session, checkpoint_path = (
            modser.load_checkpoint(checkpoint)
        )
        train_config["additional_epochs"] = additional_epochs

        train_data, validation_data = load_resume_training_data(
            train_config, model_metadata
        )

        if stop_event.is_set():
            result_queue.put({"result": {}})
            return

        logger.info("Setting device for training operations")
        device = DeviceConfig(train_config)
        device.set_device()

        logger.info("Building model data loaders")
        train_loader = XRAYDataLoader(train_config, shuffle=True).build_training_dataloader(
            train_data
        )
        validation_loader = XRAYDataLoader(
            train_config, shuffle=False
        ).build_training_dataloader(validation_data)

        trainer = ModelTrainer(train_config, model_metadata)
        reporter = QueueProgressReporter(progress_queue)
        interrupt_callback = TrainingInterruptCallback(stop_event=stop_event)
        from_epoch = session.get("epochs", 0)

        logger.info("Resuming training from epoch %s", from_epoch)
        trained_model, history = trainer.resume_training(
            model,
            train_loader,
            validation_loader,
            checkpoint_path,
            session=session,
            additional_epochs=additional_epochs,
            websocket_callback=reporter,
            interrupt_callback=interrupt_callback,
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
                    "final_val_loss": history.get("history", {}).get("val_loss", [0])[-1],
                    "checkpoint_path": checkpoint_path,
                }
            }
        )
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"error": str(exc)})
