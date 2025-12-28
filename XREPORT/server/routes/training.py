from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status

from XREPORT.server.schemas.training import (
    CheckpointInfo,
    CheckpointsResponse,
    StartTrainingRequest,
    ResumeTrainingRequest,
    TrainingStatusResponse,
)
from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.services.training.device import DeviceConfig
from XREPORT.server.utils.services.training.serializer import (
    DataSerializer,
    ModelSerializer,
)
from XREPORT.server.utils.services.training.dataloader import XRAYDataLoader
from XREPORT.server.utils.services.training.callbacks import TrainingInterruptCallback
from XREPORT.server.utils.services.training.trainer import ModelTrainer

router = APIRouter(prefix="/training", tags=["training"])

# Training state management
training_state: dict[str, Any] = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "val_loss": 0.0,
    "accuracy": 0.0,
    "val_accuracy": 0.0,
    "progress_percent": 0,
    "elapsed_seconds": 0,
    # Chart data for reconnection
    "chart_data": [],
    "epoch_boundaries": [],
    "available_metrics": [],
}

# Active WebSocket connections
active_connections: list[WebSocket] = []

# Interrupt callback for stopping training
interrupt_callback: TrainingInterruptCallback | None = None

# Thread pool for running training without blocking the event loop
training_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")

# Reference to the main event loop for WebSocket callbacks
main_event_loop: asyncio.AbstractEventLoop | None = None


# -----------------------------------------------------------------------------
async def broadcast_message(message: dict[str, Any]) -> None:
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            disconnected.append(connection)
    
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


# -----------------------------------------------------------------------------
def sync_websocket_callback(message: dict[str, Any]) -> None:
    global training_state
    
    # Update basic metrics
    training_state.update({
        "current_epoch": message.get("epoch", 0),
        "total_epochs": message.get("total_epochs", 0),
        "loss": message.get("loss", 0.0),
        "val_loss": message.get("val_loss", 0.0),
        "accuracy": message.get("accuracy", 0.0),
        "val_accuracy": message.get("val_accuracy", 0.0),
        "progress_percent": message.get("progress_percent", 0),
        "elapsed_seconds": message.get("elapsed_seconds", 0),
    })
    
    # Accumulate chart data from training_plot messages
    if message.get("type") == "training_plot":
        training_state["chart_data"] = message.get("chart_data", [])
        training_state["epoch_boundaries"] = message.get("epoch_boundaries", [])
        training_state["available_metrics"] = message.get("metrics", [])
    
    # Use the stored main event loop to schedule WebSocket broadcast
    # This works because training runs in a separate thread
    if main_event_loop is not None and main_event_loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast_message(message), main_event_loop)


###############################################################################
@router.websocket("/ws")
async def training_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(active_connections)}")
    
    try:
        # Send current state immediately
        await websocket.send_json({
            "type": "connection_established",
            "is_training": training_state["is_training"],
            **training_state,
        })
        
        # Send accumulated chart data if training is in progress
        if training_state["is_training"] and training_state["chart_data"]:
            await websocket.send_json({
                "type": "training_plot",
                "chart_data": training_state["chart_data"],
                "metrics": training_state["available_metrics"],
                "epochs": training_state["current_epoch"],
                "epoch_boundaries": training_state["epoch_boundaries"],
            })

        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket removed. Total connections: {len(active_connections)}")


###############################################################################
@router.get(
    "/checkpoints",
    response_model=CheckpointsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_checkpoints() -> CheckpointsResponse:
    """Get list of available checkpoints (JSON config only, no model loading)."""
    import os
    from XREPORT.server.utils.services.training.serializer import CHECKPOINT_PATH
    
    modser = ModelSerializer()
    checkpoint_names = modser.scan_checkpoints_folder()
    
    checkpoints = []
    for name in checkpoint_names:
        try:
            # Only load JSON configuration files, NOT the model
            checkpoint_path = os.path.join(CHECKPOINT_PATH, name)
            _, _, session = modser.load_training_configuration(checkpoint_path)
            checkpoints.append(CheckpointInfo(
                name=name,
                epochs=session.get("epochs", 0),
                loss=session.get("history", {}).get("loss", [0])[-1] if session.get("history") else 0.0,
                val_loss=session.get("history", {}).get("val_loss", [0])[-1] if session.get("history") else 0.0,
            ))
        except Exception as e:
            logger.warning(f"Failed to load checkpoint config {name}: {e}")
            checkpoints.append(CheckpointInfo(name=name, epochs=0, loss=0.0, val_loss=0.0))
    
    return CheckpointsResponse(checkpoints=checkpoints)


###############################################################################
@router.get(
    "/status",
    response_model=TrainingStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_training_status() -> TrainingStatusResponse:
    return TrainingStatusResponse(
        is_training=training_state["is_training"],
        current_epoch=training_state["current_epoch"],
        total_epochs=training_state["total_epochs"],
        loss=training_state["loss"],
        val_loss=training_state["val_loss"],
        accuracy=training_state["accuracy"],
        val_accuracy=training_state["val_accuracy"],
        progress_percent=training_state["progress_percent"],
        elapsed_seconds=training_state["elapsed_seconds"],
    )


###############################################################################
@router.post(
    "/start",
    response_model=TrainingStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def start_training(request: StartTrainingRequest) -> TrainingStatusResponse:
    global training_state, interrupt_callback, main_event_loop
    
    if training_state["is_training"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is already in progress",
        )
    
    # Build configuration from request
    configuration = request.model_dump()
    
    # Initialize serializers
    serializer = DataSerializer()
    modser = ModelSerializer()
    
    # Load training data
    train_data, validation_data, metadata = serializer.load_training_data()
    if train_data.empty or validation_data.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No training data found. Please process a dataset first.",
        )
    
    # Validate stored image paths exist
    train_data = serializer.validate_img_paths(train_data)
    validation_data = serializer.validate_img_paths(validation_data)
    
    if train_data.empty or validation_data.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid images found. Image paths may have changed since dataset was processed.",
        )
    
    # Set training state
    training_state["is_training"] = True
    training_state["current_epoch"] = 0
    training_state["total_epochs"] = configuration.get("epochs", 10)
    # Clear chart data from previous sessions
    training_state["chart_data"] = []
    training_state["epoch_boundaries"] = []
    training_state["available_metrics"] = []

    
    # Store reference to the current event loop for WebSocket callbacks
    main_event_loop = asyncio.get_running_loop()
    
    # Broadcast training started
    await broadcast_message({
        "type": "training_started",
        "total_epochs": training_state["total_epochs"],
    })
    
    # Define the blocking training function to run in a thread
    def run_training_sync() -> tuple[Any, dict[str, Any]]:
        global interrupt_callback
        
        # Set device for training
        logger.info("Setting device for training operations")
        device = DeviceConfig(configuration)
        device.set_device()
        
        # Create checkpoint folder
        checkpoint_path = modser.create_checkpoint_folder()
        
        # Build data loaders
        logger.info("Building model data loaders")
        builder = XRAYDataLoader(configuration)
        train_dataset = builder.build_training_dataloader(train_data)
        validation_dataset = builder.build_training_dataloader(validation_data)
        
        # Build model (deferred import to avoid circular imports)
        from XREPORT.server.utils.services.training.model import build_xreport_model
        
        logger.info("Building XREPORT Transformer model")
        model = build_xreport_model(metadata, configuration)
        
        # Initialize interrupt callback
        interrupt_callback = TrainingInterruptCallback()
        
        # Train model
        logger.info("Starting XREPORT Transformer model training")
        trainer = ModelTrainer(configuration)
        trained_model, history = trainer.train_model(
            model,
            train_dataset,
            validation_dataset,
            checkpoint_path,
            websocket_callback=sync_websocket_callback,
            interrupt_callback=interrupt_callback,
        )
        
        # Save model and configuration
        modser.save_pretrained_model(trained_model, checkpoint_path)
        modser.save_training_configuration(checkpoint_path, history, configuration, metadata)
        
        return trained_model, history
    
    try:
        # Run training in a thread so the event loop stays free for WebSocket
        loop = asyncio.get_running_loop()
        model, history = await loop.run_in_executor(training_executor, run_training_sync)
        
        # Broadcast training completed
        await broadcast_message({
            "type": "training_completed",
            "epochs": history.get("epochs", 0),
            "final_loss": history.get("history", {}).get("loss", [0])[-1],
            "final_val_loss": history.get("history", {}).get("val_loss", [0])[-1],
        })
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.exception("Training failed")
        await broadcast_message({
            "type": "training_error",
            "error": str(e),
        })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        ) from e
    
    finally:        
        training_state["is_training"] = False
        interrupt_callback = None
    
    return TrainingStatusResponse(
        is_training=False,
        current_epoch=training_state["current_epoch"],
        total_epochs=training_state["total_epochs"],
        loss=training_state["loss"],
        val_loss=training_state["val_loss"],
        accuracy=training_state["accuracy"],
        val_accuracy=training_state["val_accuracy"],
        progress_percent=100,
        elapsed_seconds=training_state["elapsed_seconds"],
    )


###############################################################################
@router.post(
    "/resume",
    response_model=TrainingStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def resume_training(request: ResumeTrainingRequest) -> TrainingStatusResponse:
    global training_state, interrupt_callback, main_event_loop
    
    if training_state["is_training"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is already in progress",
        )
    
    # Initialize serializers
    serializer = DataSerializer()
    modser = ModelSerializer()
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {request.checkpoint}")
    try:
        model, train_config, model_metadata, session, checkpoint_path = modser.load_checkpoint(
            request.checkpoint
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Checkpoint not found: {request.checkpoint}",
        ) from e
    
    # Load and validate training data
    current_metadata = serializer.load_training_data(only_metadata=True)
    is_validated = serializer.validate_metadata(current_metadata, model_metadata)
    
    if is_validated:
        logger.info("Loading processed dataset")
        train_data, validation_data, _ = serializer.load_training_data()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current dataset metadata doesn't match checkpoint. Please reprocess the dataset.",
        )
    
    # Validate stored image paths exist
    train_data = serializer.validate_img_paths(train_data)
    validation_data = serializer.validate_img_paths(validation_data)
    
    if train_data.empty or validation_data.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid images found. Image paths may have changed since dataset was processed.",
        )
    
    # Set training state
    from_epoch = session.get("epochs", 0)
    training_state["is_training"] = True
    training_state["current_epoch"] = from_epoch
    training_state["total_epochs"] = from_epoch + request.additional_epochs
    
    # Store reference to the current event loop for WebSocket callbacks
    main_event_loop = asyncio.get_running_loop()
    
    # Broadcast training resumed
    await broadcast_message({
        "type": "training_resumed",
        "from_epoch": from_epoch,
        "additional_epochs": request.additional_epochs,
    })
    
    # Define the blocking resume function to run in a thread
    def run_resume_sync() -> tuple[Any, dict[str, Any]]:
        global interrupt_callback
        
        # Set device for training
        logger.info("Setting device for training operations")
        device = DeviceConfig(train_config)
        device.set_device()
        
        # Build data loaders
        logger.info("Building model data loaders")
        builder = XRAYDataLoader(train_config)
        train_dataset = builder.build_training_dataloader(train_data)
        validation_dataset = builder.build_training_dataloader(validation_data)
        
        # Initialize interrupt callback
        interrupt_callback = TrainingInterruptCallback()
        
        # Resume training
        logger.info(f"Resuming training from epoch {from_epoch}")
        trainer = ModelTrainer(train_config, model_metadata)
        trained_model, history = trainer.resume_training(
            model,
            train_dataset,
            validation_dataset,
            checkpoint_path,
            session=session,
            additional_epochs=request.additional_epochs,
            websocket_callback=sync_websocket_callback,
            interrupt_callback=interrupt_callback,
        )
        
        # Save model and configuration
        modser.save_pretrained_model(trained_model, checkpoint_path)
        modser.save_training_configuration(checkpoint_path, history, train_config, model_metadata)
        
        return trained_model, history
    
    try:
        # Run training in a thread so the event loop stays free for WebSocket
        loop = asyncio.get_running_loop()
        model, history = await loop.run_in_executor(training_executor, run_resume_sync)
        
        # Broadcast training completed
        await broadcast_message({
            "type": "training_completed",
            "epochs": history.get("epochs", 0),
            "final_loss": history.get("history", {}).get("loss", [0])[-1],
            "final_val_loss": history.get("history", {}).get("val_loss", [0])[-1],
        })
        
        logger.info("Resumed training completed successfully")
        
    except Exception as e:
        logger.exception("Resume training failed")
        await broadcast_message({
            "type": "training_error",
            "error": str(e),
        })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        ) from e
    
    finally:
        training_state["is_training"] = False
        interrupt_callback = None
    
    return TrainingStatusResponse(
        is_training=False,
        current_epoch=training_state["current_epoch"],
        total_epochs=training_state["total_epochs"],
        loss=training_state["loss"],
        val_loss=training_state["val_loss"],
        accuracy=training_state["accuracy"],
        val_accuracy=training_state["val_accuracy"],
        progress_percent=100,
        elapsed_seconds=training_state["elapsed_seconds"],
    )


###############################################################################
@router.post(
    "/stop",
    response_model=TrainingStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def stop_training() -> TrainingStatusResponse:
    global interrupt_callback
    
    if not training_state["is_training"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No training is currently in progress",
        )
    
    if interrupt_callback is not None:
        interrupt_callback.request_stop()
        logger.info("Training stop requested")
        
        await broadcast_message({
            "type": "training_stopping",
            "message": "Training stop requested. Will stop after current epoch.",
        })
    
    return TrainingStatusResponse(
        is_training=training_state["is_training"],
        current_epoch=training_state["current_epoch"],
        total_epochs=training_state["total_epochs"],
        loss=training_state["loss"],
        val_loss=training_state["val_loss"],
        accuracy=training_state["accuracy"],
        val_accuracy=training_state["val_accuracy"],
        progress_percent=training_state["progress_percent"],
        elapsed_seconds=training_state["elapsed_seconds"],
    )
