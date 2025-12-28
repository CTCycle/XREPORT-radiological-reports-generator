from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from XREPORT.server.utils.constants import RESOURCES_PATH
from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.services.inference import TextGenerator
from XREPORT.server.utils.services.training.serializer import ModelSerializer

router = APIRouter(prefix="/inference", tags=["inference"])

# WebSocket connections for inference streaming
inference_connections: list[WebSocket] = []

# Thread pool for inference without blocking event loop
inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

# Main event loop reference for WebSocket callbacks
main_event_loop: asyncio.AbstractEventLoop | None = None

# Inference temp folder for uploaded images
INFERENCE_TEMP_PATH = os.path.join(RESOURCES_PATH, "inference_temp")


###############################################################################
class CheckpointInfo(BaseModel):
    name: str
    created: str | None = None


class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointInfo]
    success: bool
    message: str


class GenerationRequest(BaseModel):
    checkpoint: str
    generation_mode: str


class GenerationResponse(BaseModel):
    success: bool
    message: str
    reports: dict[str, str] | None = None


# -----------------------------------------------------------------------------
def broadcast_inference_message(message: dict[str, Any]) -> None:
    global main_event_loop
    if main_event_loop is None:
        logger.warning("No event loop available for WebSocket broadcast")
        return
    if len(inference_connections) == 0:
        logger.warning("No WebSocket connections to broadcast to")
        return

    async def send_to_all() -> None:
        disconnected = []
        for connection in inference_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            if conn in inference_connections:
                inference_connections.remove(conn)

    asyncio.run_coroutine_threadsafe(send_to_all(), main_event_loop)


# -----------------------------------------------------------------------------
def sync_stream_callback(
    image_idx: int, token: str, step: int, total: int
) -> None:
    logger.debug(f"Streaming token {step}/{total}: {token}")
    broadcast_inference_message({
        "type": "token",
        "image_index": image_idx,
        "token": token,
        "step": step,
        "total": total,
    })


###############################################################################
@router.websocket("/ws")
async def inference_websocket(websocket: WebSocket) -> None:
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()

    await websocket.accept()
    inference_connections.append(websocket)
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
        if websocket in inference_connections:
            inference_connections.remove(websocket)


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


###############################################################################
@router.post("/generate")
async def generate_reports(
    checkpoint: str = Form(...),
    generation_mode: str = Form(...),
    images: list[UploadFile] = File(...),
) -> GenerationResponse:
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()

    if len(images) == 0:
        return GenerationResponse(
            success=False,
            message="No images provided",
        )

    if len(images) > 16:
        return GenerationResponse(
            success=False,
            message="Maximum 16 images allowed",
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

        # Broadcast generation start
        broadcast_inference_message({
            "type": "start",
            "total_images": len(image_paths),
            "checkpoint": checkpoint,
            "mode": generation_mode,
        })

        # Load model checkpoint
        logger.info(f"Loading checkpoint: {checkpoint}")
        serializer = ModelSerializer()
        model, train_config, model_metadata, _, _ = serializer.load_checkpoint(
            checkpoint
        )
        model.summary(expand_nested=True)

        max_report_size = model_metadata.get("max_report_size", 200)

        # Initialize generator with model metadata (contains tokenizer info)
        generator = TextGenerator(model, model_metadata, max_report_size)

        # Run generation in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        reports = await loop.run_in_executor(
            inference_executor,
            lambda: generator.generate_radiological_reports(
                image_paths,
                generation_mode,
                stream_callback=sync_stream_callback,
            ),
        )

        if reports is None:
            broadcast_inference_message({
                "type": "error",
                "message": "Failed to generate reports",
            })
            return GenerationResponse(
                success=False,
                message="Failed to generate reports",
            )

        # Convert paths to filenames for response
        reports_by_filename = {
            os.path.basename(k): v for k, v in reports.items()
        }

        broadcast_inference_message({
            "type": "complete",
            "reports": reports_by_filename,
        })

        return GenerationResponse(
            success=True,
            message=f"Generated {len(reports)} reports",
            reports=reports_by_filename,
        )

    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        broadcast_inference_message({
            "type": "error",
            "message": str(e),
        })
        return GenerationResponse(
            success=False,
            message=str(e),
        )

    finally:
        # Clean up temp files
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
