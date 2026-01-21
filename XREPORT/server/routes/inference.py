from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from XREPORT.server.utils.constants import RESOURCES_PATH
from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.services.inference import TextGenerator
from XREPORT.server.utils.services.training.serializer import ModelSerializer

router = APIRouter(prefix="/inference", tags=["inference"])

# Inference temp folder for uploaded images
INFERENCE_TEMP_PATH = os.path.join(RESOURCES_PATH, "inference_temp")


###############################################################################
class InferenceState:
    """Encapsulates all inference session state."""

    def __init__(self) -> None:
        self.connections: list[WebSocket] = []
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="inference"
        )
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


###############################################################################
@router.post("/generate", response_model=GenerationResponse)
async def generate_reports(
    checkpoint: str = Form(...),
    generation_mode: str = Form(...),
    images: list[UploadFile] = File(...),
) -> GenerationResponse:
    inference_state.event_loop = asyncio.get_event_loop()

    if len(images) == 0:
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "No images provided",
        )

    if len(images) > 16:
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "Maximum 16 images allowed",
        )

    allowed_modes = {"greedy_search", "beam_search"}
    if generation_mode not in allowed_modes:
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported generation mode: {generation_mode}",
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
        inference_state.broadcast({
            "type": "start",
            "total_images": len(image_paths),
            "checkpoint": checkpoint,
            "mode": generation_mode,
        })

        # Load model checkpoint
        logger.info(f"Loading checkpoint: {checkpoint}")
        serializer = ModelSerializer()
        try:
            model, train_config, model_metadata, _, _ = serializer.load_checkpoint(
                checkpoint
            )
        except Exception as e:
            logger.error(f"Checkpoint load failed for {checkpoint}: {e}")
            return error_response(
                status.HTTP_404_NOT_FOUND,
                f"Checkpoint not found: {checkpoint}",
            )
        model.summary(expand_nested=True)

        max_report_size = model_metadata.get("max_report_size", 200)

        # Initialize generator with model metadata (contains tokenizer info)
        generator = TextGenerator(model, model_metadata, max_report_size)

        # Run generation in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        reports = await loop.run_in_executor(
            inference_state.executor,
            lambda: generator.generate_radiological_reports(
                image_paths,
                generation_mode,
                stream_callback=inference_state.stream_token,
            ),
        )

        if reports is None:
            inference_state.broadcast({
                "type": "error",
                "message": "Failed to generate reports",
            })
            return error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Failed to generate reports",
            )

        # Convert paths to filenames for response
        reports_by_filename = {
            os.path.basename(k): v for k, v in reports.items()
        }

        inference_state.broadcast({
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
        inference_state.broadcast({
            "type": "error",
            "message": str(e),
        })
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            str(e),
        )

    finally:
        # Clean up temp files
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
