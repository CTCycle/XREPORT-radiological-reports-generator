from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from XREPORT.server.schemas.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from XREPORT.server.utils.constants import RESOURCES_PATH
from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.services.inference import TextGenerator
from XREPORT.server.utils.services.jobs import job_manager
from XREPORT.server.utils.services.training.serializer import ModelSerializer

router = APIRouter(prefix="/inference", tags=["inference"])

# Inference temp folder for uploaded images
INFERENCE_TEMP_PATH = os.path.join(RESOURCES_PATH, "inference_temp")


###############################################################################
class InferenceState:
    """Encapsulates all inference session state."""

    def __init__(self) -> None:
        self.connections: list[WebSocket] = []
        self.event_loop: asyncio.AbstractEventLoop | None = None

    # -------------------------------------------------------------------------
    def broadcast(self, message: dict[str, Any]) -> None:
        """Thread-safe broadcast to all connected clients."""
        if self.event_loop is None:
            logger.warning("No event loop available for WebSocket broadcast")
            return
        if len(self.connections) == 0:
            logger.warning("No WebSocket connections to broadcast to")
            return

        async def send_to_all() -> None:
            disconnected = []
            for connection in self.connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            for conn in disconnected:
                if conn in self.connections:
                    self.connections.remove(conn)

        asyncio.run_coroutine_threadsafe(send_to_all(), self.event_loop)

    # -------------------------------------------------------------------------
    def add_connection(self, websocket: WebSocket) -> None:
        self.connections.append(websocket)

    # -------------------------------------------------------------------------
    def remove_connection(self, websocket: WebSocket) -> None:
        if websocket in self.connections:
            self.connections.remove(websocket)

    # -------------------------------------------------------------------------
    def stream_token(self, image_idx: int, token: str, step: int, total: int) -> None:
        """Stream a generated token to all connected clients."""
        logger.debug(f"Streaming token {step}/{total}: {token}")
        self.broadcast({
            "type": "token",
            "image_index": image_idx,
            "token": token,
            "step": step,
            "total": total,
        })


inference_state = InferenceState()


###############################################################################
class CheckpointInfo(BaseModel):
    name: str
    created: str | None = None


###############################################################################
class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointInfo]
    success: bool
    message: str


###############################################################################
class GenerationRequest(BaseModel):
    checkpoint: str
    generation_mode: str


###############################################################################
class GenerationResponse(BaseModel):
    success: bool
    message: str
    reports: dict[str, str] | None = None


# -----------------------------------------------------------------------------
def error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=GenerationResponse(
            success=False,
            message=message,
            reports=None,
        ).model_dump(),
    )


###############################################################################
@router.websocket("/ws")
async def inference_websocket(websocket: WebSocket) -> None:
    inference_state.event_loop = asyncio.get_event_loop()

    await websocket.accept()
    inference_state.add_connection(websocket)
    logger.info("Inference WebSocket connection established")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Inference WebSocket connection closed")
    except Exception as e:
        logger.error(f"Inference WebSocket error: {e}")
    finally:
        inference_state.remove_connection(websocket)


###############################################################################
@router.get("/checkpoints", response_model=CheckpointsResponse)
def get_checkpoints() -> CheckpointsResponse:
    try:
        serializer = ModelSerializer()
        checkpoint_names = serializer.scan_checkpoints_folder()

        checkpoints = [
            CheckpointInfo(name=name, created=None) for name in checkpoint_names
        ]

        return CheckpointsResponse(
            checkpoints=checkpoints,
            success=True,
            message=f"Found {len(checkpoints)} checkpoints",
        )
    except Exception as e:
        logger.error(f"Error fetching checkpoints: {e}")
        return CheckpointsResponse(
            checkpoints=[],
            success=False,
            message=str(e),
        )


# -----------------------------------------------------------------------------
def run_inference_job(
    checkpoint: str,
    generation_mode: str,
    image_paths: list[str],
    job_id: str,
) -> dict[str, Any]:
    """Blocking inference function that runs in background thread."""
    # Load model checkpoint
    logger.info(f"Loading checkpoint: {checkpoint}")
    serializer = ModelSerializer()
    
    try:
        model, train_config, model_metadata, _, _ = serializer.load_checkpoint(checkpoint)
    except Exception as e:
        logger.error(f"Checkpoint load failed for {checkpoint}: {e}")
        raise RuntimeError(f"Checkpoint not found: {checkpoint}") from e
    
    model.summary(expand_nested=True)
    max_report_size = model_metadata.get("max_report_size", 200)

    # Initialize generator with model metadata (contains tokenizer info)
    generator = TextGenerator(model, model_metadata, max_report_size)

    # Broadcast generation start
    inference_state.broadcast({
        "type": "start",
        "job_id": job_id,
        "total_images": len(image_paths),
        "checkpoint": checkpoint,
        "mode": generation_mode,
    })

    # Generate reports
    reports = generator.generate_radiological_reports(
        image_paths,
        generation_mode,
        stream_callback=inference_state.stream_token,
    )

    if reports is None:
        inference_state.broadcast({
            "type": "error",
            "job_id": job_id,
            "message": "Failed to generate reports",
        })
        raise RuntimeError("Failed to generate reports")

    # Convert paths to filenames for response
    reports_by_filename = {
        os.path.basename(k): v for k, v in reports.items()
    }

    inference_state.broadcast({
        "type": "complete",
        "job_id": job_id,
        "reports": reports_by_filename,
    })

    # Clean up temp files
    for path in image_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    return {
        "reports": reports_by_filename,
        "count": len(reports_by_filename),
    }


###############################################################################
@router.post("/generate", response_model=JobStartResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_reports(
    checkpoint: str = Form(...),
    generation_mode: str = Form(...),
    images: list[UploadFile] = File(...),
) -> JobStartResponse:
    inference_state.event_loop = asyncio.get_event_loop()

    if len(images) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No images provided",
        )

    if len(images) > 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 16 images allowed",
        )

    allowed_modes = {"greedy_search", "beam_search"}
    if generation_mode not in allowed_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported generation mode: {generation_mode}",
        )

    # Create temp directory for uploaded images
    os.makedirs(INFERENCE_TEMP_PATH, exist_ok=True)

    image_paths: list[str] = []
    try:
        # Save uploaded images to temp location
        for img in images:
            if img.filename is None:
                continue
            temp_path = os.path.join(INFERENCE_TEMP_PATH, img.filename)
            content = await img.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            image_paths.append(temp_path)

        # Start background job
        job_id = job_manager.start_job(
            job_type="inference",
            runner=run_inference_job,
            kwargs={
                "checkpoint": checkpoint,
                "generation_mode": generation_mode,
                "image_paths": image_paths,
                "job_id": "",
            },
        )

        return JobStartResponse(
            job_id=job_id,
            message=f"Inference job started for {len(image_paths)} images",
        )

    except Exception as e:
        # Clean up on error
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        logger.error(f"Error starting inference job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


###############################################################################
@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_inference_job_status(job_id: str) -> JobStatusResponse:
    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return JobStatusResponse(**job_status)


###############################################################################
@router.delete(
    "/jobs/{job_id}",
    response_model=JobCancelResponse,
    status_code=status.HTTP_200_OK,
)
async def cancel_inference_job(job_id: str) -> JobCancelResponse:
    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    
    success = job_manager.cancel_job(job_id)
    
    return JobCancelResponse(
        job_id=job_id,
        success=success,
        message="Cancellation requested" if success else "Job cannot be cancelled",
    )


# Import HTTPException at module level
from fastapi import HTTPException
