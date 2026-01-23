from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status

from XREPORT.server.schemas.training import (
    CheckpointInfo,
    CheckpointsResponse,
    StartTrainingRequest,
    ResumeTrainingRequest,
    TrainingStatusResponse,
)
from XREPORT.server.schemas.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.services.jobs import JobManager, job_manager
from XREPORT.server.utils.configurations.server import ServerSettings, server_settings
from XREPORT.server.utils.services.training.device import DeviceConfig
from XREPORT.server.utils.services.training.serializer import (
    DataSerializer,
    ModelSerializer,
    CHECKPOINT_PATH,
)
from XREPORT.server.utils.services.training.dataloader import XRAYDataLoader
from XREPORT.server.utils.services.training.callbacks import TrainingInterruptCallback
from XREPORT.server.utils.services.training.trainer import ModelTrainer
# Moved imports to top-level as requested
from XREPORT.server.utils.services.training.model import build_xreport_model


###############################################################################
class TrainingState:
    """Encapsulates all training session state."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {
            "is_training": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "loss": 0.0,
            "val_loss": 0.0,
            "accuracy": 0.0,
            "val_accuracy": 0.0,
            "progress_percent": 0,
            "elapsed_seconds": 0,
            "chart_data": [],
            "epoch_boundaries": [],
            "available_metrics": [],
        }
        self.connections: list[WebSocket] = []
        self.interrupt_callback: TrainingInterruptCallback | None = None
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.current_job_id: str | None = None

    # -----------------------------------------------------------------------------
    def update_metrics(self, message: dict[str, Any]) -> None:
        """Update state from a training callback message."""
        self.state.update({
            "current_epoch": message.get("epoch", 0),
            "total_epochs": message.get("total_epochs", 0),
            "loss": message.get("loss", 0.0),
            "val_loss": message.get("val_loss", 0.0),
            "accuracy": message.get("accuracy", 0.0),
            "val_accuracy": message.get("val_accuracy", 0.0),
            "progress_percent": message.get("progress_percent", 0),
            "elapsed_seconds": message.get("elapsed_seconds", 0),
        })
        if message.get("type") == "training_plot":
            self.state["chart_data"] = message.get("chart_data", [])
            self.state["epoch_boundaries"] = message.get("epoch_boundaries", [])
            self.state["available_metrics"] = message.get("metrics", [])

    # -----------------------------------------------------------------------------
    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected WebSocket clients."""
        disconnected = []
        for connection in self.connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            if conn in self.connections:
                self.connections.remove(conn)

    # -----------------------------------------------------------------------------
    def sync_broadcast(self, message: dict[str, Any]) -> None:
        """Thread-safe broadcast from training thread."""
        self.update_metrics(message)
        if self.event_loop is not None and self.event_loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.event_loop)

    # -----------------------------------------------------------------------------
    def add_connection(self, websocket: WebSocket) -> None:
        self.connections.append(websocket)

    # -----------------------------------------------------------------------------
    def remove_connection(self, websocket: WebSocket) -> None:
        if websocket in self.connections:
            self.connections.remove(websocket)

    # -----------------------------------------------------------------------------
    def reset_for_new_session(self, total_epochs: int, job_id: str) -> None:
        """Reset state for a new training session."""
        self.state["is_training"] = True
        self.state["current_epoch"] = 0
        self.state["total_epochs"] = total_epochs
        self.state["chart_data"] = []
        self.state["epoch_boundaries"] = []
        self.state["available_metrics"] = []
        self.current_job_id = job_id

    # -----------------------------------------------------------------------------
    def finish_session(self) -> None:
        """Mark training session as complete."""
        self.state["is_training"] = False
        self.interrupt_callback = None
        self.current_job_id = None


training_state = TrainingState()


# -----------------------------------------------------------------------------
def run_training_job(
    configuration: dict[str, Any],
    train_data: Any,
    validation_data: Any,
    metadata: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Blocking training function that runs in background thread."""
    # Note: Accessing training_state directly via module-level import to support ThreadPoolExecutor
    # without pickle issues.
    
    # Initialize serializers
    modser = ModelSerializer()
    
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
    
    logger.info("Building XREPORT Transformer model")
    model = build_xreport_model(metadata, configuration)
    
    # Initialize interrupt callback that checks job manager
    training_state.interrupt_callback = TrainingInterruptCallback()
    
    # Train model
    logger.info("Starting XREPORT Transformer model training")
    trainer = ModelTrainer(configuration)
    trained_model, history = trainer.train_model(
        model,
        train_dataset,
        validation_dataset,
        checkpoint_path,
        websocket_callback=training_state.sync_broadcast,
        interrupt_callback=training_state.interrupt_callback,
    )
    
    # Save model and configuration
    modser.save_pretrained_model(trained_model, checkpoint_path)
    modser.save_training_configuration(checkpoint_path, history, configuration, metadata)
    
    # Broadcast training completed
    if training_state.event_loop is not None and training_state.event_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            training_state.broadcast({
                "type": "training_completed",
                "epochs": history.get("epochs", 0),
                "final_loss": history.get("history", {}).get("loss", [0])[-1],
                "final_val_loss": history.get("history", {}).get("val_loss", [0])[-1],
            }),
            training_state.event_loop,
        )
    
    training_state.finish_session()
    
    return {
        "epochs": history.get("epochs", 0),
        "final_loss": history.get("history", {}).get("loss", [0])[-1],
        "final_val_loss": history.get("history", {}).get("val_loss", [0])[-1],
        "checkpoint_path": checkpoint_path,
    }


# -----------------------------------------------------------------------------
def run_resume_training_job(
    model: Any,
    train_config: dict[str, Any],
    model_metadata: dict[str, Any],
    session: dict[str, Any],
    checkpoint_path: str,
    train_data: Any,
    validation_data: Any,
    additional_epochs: int,
    job_id: str,
) -> dict[str, Any]:
    """Blocking resume training function that runs in background thread."""
    modser = ModelSerializer()
    from_epoch = session.get("epochs", 0)
    
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
    training_state.interrupt_callback = TrainingInterruptCallback()
    
    # Resume training
    logger.info(f"Resuming training from epoch {from_epoch}")
    trainer = ModelTrainer(train_config, model_metadata)
    trained_model, history = trainer.resume_training(
        model,
        train_dataset,
        validation_dataset,
        checkpoint_path,
        session=session,
        additional_epochs=additional_epochs,
        websocket_callback=training_state.sync_broadcast,
        interrupt_callback=training_state.interrupt_callback,
    )
    
    # Save model and configuration
    modser.save_pretrained_model(trained_model, checkpoint_path)
    modser.save_training_configuration(checkpoint_path, history, train_config, model_metadata)
    
    # Broadcast training completed
    if training_state.event_loop is not None and training_state.event_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            training_state.broadcast({
                "type": "training_completed",
                "epochs": history.get("epochs", 0),
                "final_loss": history.get("history", {}).get("loss", [0])[-1],
                "final_val_loss": history.get("history", {}).get("val_loss", [0])[-1],
            }),
            training_state.event_loop,
        )
    
    training_state.finish_session()
    
    return {
        "epochs": history.get("epochs", 0),
        "final_loss": history.get("history", {}).get("loss", [0])[-1],
        "final_val_loss": history.get("history", {}).get("val_loss", [0])[-1],
        "checkpoint_path": checkpoint_path,
    }


###############################################################################
class TrainingEndpoint:
    """Endpoint for model training operations."""

    def __init__(
        self,
        router: APIRouter,
        job_manager: JobManager,
        training_state: TrainingState,
    ) -> None:
        self.router = router
        self.job_manager = job_manager
        self.training_state = training_state

    # -----------------------------------------------------------------------------
    async def training_websocket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.training_state.add_connection(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.training_state.connections)}")
        
        try:
            # Send current state immediately
            await websocket.send_json({
                "type": "connection_established",
                "is_training": self.training_state.state["is_training"],
                **self.training_state.state,
            })
            
            # Send accumulated chart data if training is in progress
            if self.training_state.state["is_training"] and self.training_state.state["chart_data"]:
                await websocket.send_json({
                    "type": "training_plot",
                    "chart_data": self.training_state.state["chart_data"],
                    "metrics": self.training_state.state["available_metrics"],
                    "epochs": self.training_state.state["current_epoch"],
                    "epoch_boundaries": self.training_state.state["epoch_boundaries"],
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
            self.training_state.remove_connection(websocket)
            logger.info(f"WebSocket removed. Total connections: {len(self.training_state.connections)}")

    # -----------------------------------------------------------------------------
    async def get_checkpoints(self) -> CheckpointsResponse:
        """Get list of available checkpoints (JSON config only, no model loading)."""
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

    # -----------------------------------------------------------------------------
    async def get_training_status(self) -> TrainingStatusResponse:
        return TrainingStatusResponse(
            is_training=self.training_state.state["is_training"],
            current_epoch=self.training_state.state["current_epoch"],
            total_epochs=self.training_state.state["total_epochs"],
            loss=self.training_state.state["loss"],
            val_loss=self.training_state.state["val_loss"],
            accuracy=self.training_state.state["accuracy"],
            val_accuracy=self.training_state.state["val_accuracy"],
            progress_percent=self.training_state.state["progress_percent"],
            elapsed_seconds=self.training_state.state["elapsed_seconds"],
        )

    # -----------------------------------------------------------------------------
    async def start_training(self, request: StartTrainingRequest) -> JobStartResponse:
        if self.job_manager.is_job_running("training"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Training is already in progress",
            )
        
        # Build configuration from request
        configuration = request.model_dump()
        
        # Initialize serializers
        serializer = DataSerializer()
        
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
        
        # Store reference to the current event loop for WebSocket callbacks
        self.training_state.event_loop = asyncio.get_running_loop()
        
        # Start background job
        job_id = self.job_manager.start_job(
            job_type="training",
            runner=run_training_job,
            kwargs={
                "configuration": configuration,
                "train_data": train_data,
                "validation_data": validation_data,
                "metadata": metadata,
                "job_id": "",  # Will be updated below
            },
        )
        
        # Update the job_id in the kwargs (hacky but necessary)
        with self.job_manager.lock:
            state = self.job_manager.jobs.get(job_id)
            if state:
                self.training_state.reset_for_new_session(configuration.get("epochs", 10), job_id)
        
        # Broadcast training started
        await self.training_state.broadcast({
            "type": "training_started",
            "job_id": job_id,
            "total_epochs": configuration.get("epochs", 10),
        })
        
        return JobStartResponse(
            job_id=job_id,
            message="Training job started",
        )

    # -----------------------------------------------------------------------------
    async def resume_training(self, request: ResumeTrainingRequest) -> JobStartResponse:
        if self.job_manager.is_job_running("training"):
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
        
        from_epoch = session.get("epochs", 0)
        
        # Store reference to the current event loop for WebSocket callbacks
        self.training_state.event_loop = asyncio.get_running_loop()
        
        # Start background job
        job_id = self.job_manager.start_job(
            job_type="training",
            runner=run_resume_training_job,
            kwargs={
                "model": model,
                "train_config": train_config,
                "model_metadata": model_metadata,
                "session": session,
                "checkpoint_path": checkpoint_path,
                "train_data": train_data,
                "validation_data": validation_data,
                "additional_epochs": request.additional_epochs,
                "job_id": "",
            },
        )
        
        # Update training state
        self.training_state.reset_for_new_session(from_epoch + request.additional_epochs, job_id)
        self.training_state.state["current_epoch"] = from_epoch
        
        # Broadcast training resumed
        await self.training_state.broadcast({
            "type": "training_resumed",
            "job_id": job_id,
            "from_epoch": from_epoch,
            "additional_epochs": request.additional_epochs,
        })
        
        return JobStartResponse(
            job_id=job_id,
            message=f"Training resumed from epoch {from_epoch}",
        )

    # -----------------------------------------------------------------------------
    async def get_training_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return JobStatusResponse(**job_status)

    # -----------------------------------------------------------------------------
    async def cancel_training_job(self, job_id: str) -> JobCancelResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        
        # Also trigger the interrupt callback if training is active
        if self.training_state.interrupt_callback is not None:
            self.training_state.interrupt_callback.request_stop()
        
        success = self.job_manager.cancel_job(job_id)
        
        if success:
            await self.training_state.broadcast({
                "type": "training_stopping",
                "job_id": job_id,
                "message": "Training stop requested. Will stop after current epoch.",
            })
        
        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )

    # -----------------------------------------------------------------------------
    async def stop_training(self) -> TrainingStatusResponse:
        """Legacy stop endpoint - maintained for backward compatibility."""
        if not self.training_state.state["is_training"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No training is currently in progress",
            )
        
        if self.training_state.interrupt_callback is not None:
            self.training_state.interrupt_callback.request_stop()
            logger.info("Training stop requested")
            
            # Also cancel via job manager if we have a job_id
            if self.training_state.current_job_id:
                self.job_manager.cancel_job(self.training_state.current_job_id)
            
            await self.training_state.broadcast({
                "type": "training_stopping",
                "message": "Training stop requested. Will stop after current epoch.",
            })
        
        return TrainingStatusResponse(
            is_training=self.training_state.state["is_training"],
            current_epoch=self.training_state.state["current_epoch"],
            total_epochs=self.training_state.state["total_epochs"],
            loss=self.training_state.state["loss"],
            val_loss=self.training_state.state["val_loss"],
            accuracy=self.training_state.state["accuracy"],
            val_accuracy=self.training_state.state["val_accuracy"],
            progress_percent=self.training_state.state["progress_percent"],
            elapsed_seconds=self.training_state.state["elapsed_seconds"],
        )

    # -----------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all training-related routes."""
        self.router.add_api_websocket_route("/ws", self.training_websocket)
        self.router.add_api_route(
            "/checkpoints",
            self.get_checkpoints,
            methods=["GET"],
            response_model=CheckpointsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/status",
            self.get_training_status,
            methods=["GET"],
            response_model=TrainingStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/start",
            self.start_training,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/resume",
            self.resume_training,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_training_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_training_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/stop",
            self.stop_training,
            methods=["POST"],
            response_model=TrainingStatusResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
router = APIRouter(prefix="/training", tags=["training"])
training_endpoint = TrainingEndpoint(
    router=router,
    job_manager=job_manager,
    training_state=training_state,
)
training_endpoint.add_routes()
