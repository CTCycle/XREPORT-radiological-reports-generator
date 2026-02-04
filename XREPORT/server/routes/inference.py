from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile, status, HTTPException
from fastapi.responses import JSONResponse

from XREPORT.server.schemas.inference import (
    CheckpointInfo,
    CheckpointsResponse,
    GenerationResponse,
)
from XREPORT.server.schemas.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from XREPORT.server.utils.constants import CHECKPOINT_PATH
from XREPORT.server.utils.logger import logger
from XREPORT.server.learning.inference import TextGenerator
from XREPORT.server.learning.training.dataloader import XRAYDataLoader
from XREPORT.server.services.jobs import JobManager, job_manager
from XREPORT.server.repositories.serializer import ModelSerializer


MAX_INFERENCE_IMAGES = 16
MAX_TOTAL_IMAGE_BYTES = 64 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
    "image/tiff",
}
ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


###############################################################################
@dataclass(frozen=True)
class InferenceImage:
    filename: str
    content_type: str
    data: bytes
    size_bytes: int


###############################################################################
class InferenceImageStore:
    def __init__(self) -> None:
        self.storage: dict[str, list[InferenceImage]] = {}
        self.job_links: dict[str, str] = {}
        self.lock = threading.Lock()

    # -------------------------------------------------------------------------
    def store(self, request_id: str, images: list[InferenceImage]) -> None:
        with self.lock:
            self.storage[request_id] = images

    # -------------------------------------------------------------------------
    def get(self, request_id: str) -> list[InferenceImage] | None:
        with self.lock:
            return self.storage.get(request_id)

    # -------------------------------------------------------------------------
    def remove_request(self, request_id: str) -> None:
        with self.lock:
            self.storage.pop(request_id, None)

    # -------------------------------------------------------------------------
    def link_job(self, job_id: str, request_id: str) -> None:
        with self.lock:
            self.job_links[job_id] = request_id

    # -------------------------------------------------------------------------
    def remove_job(self, job_id: str) -> None:
        with self.lock:
            request_id = self.job_links.pop(job_id, None)
            if request_id is None:
                return
            self.storage.pop(request_id, None)


inference_image_store = InferenceImageStore()


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


# -----------------------------------------------------------------------------
def run_inference_job(
    checkpoint: str,
    generation_mode: str,
    request_id: str,
    job_id: str,
) -> dict[str, Any]:
    """Blocking inference function that runs in background thread."""
    if job_manager.should_stop(job_id):
        inference_image_store.remove_job(job_id)
        return {}

    stored_images = inference_image_store.get(request_id)
    if stored_images is None or len(stored_images) == 0:
        logger.error("Inference job %s has no images to process", job_id)
        raise RuntimeError("No images available for inference job")

    # Load model checkpoint
    logger.info(f"Loading checkpoint: {checkpoint}")
    serializer = ModelSerializer()

    try:
        model, train_config, model_metadata, _, _ = serializer.load_checkpoint(
            checkpoint
        )
    except Exception as e:
        logger.error(f"Checkpoint load failed for {checkpoint}: {e}")
        raise RuntimeError(f"Checkpoint not found: {checkpoint}") from e

    model.summary(expand_nested=True)
    max_report_size = model_metadata.get("max_report_size", 200)

    # Initialize generator with model metadata (contains tokenizer info)
    generator = TextGenerator(model, model_metadata, max_report_size)
    tokenizers_info = generator.load_tokenizer_and_configuration()
    if tokenizers_info is None:
        raise RuntimeError("Failed to load tokenizer")

    tokenizer, tokenizer_config = tokenizers_info
    vocabulary = tokenizer.get_vocab()

    generator_fn = generator.generator_image_methods.get(generation_mode)
    if generator_fn is None:
        raise RuntimeError(f"Unknown generation mode: {generation_mode}")

    reports_by_filename: dict[str, str] = {}
    total_images = len(stored_images)
    dataloader = XRAYDataLoader(model_metadata, shuffle=False)

    try:
        for image_index, stored_image in enumerate(stored_images, start=1):
            if job_manager.should_stop(job_id):
                break

            try:
                image = dataloader.prepare_inference_image_bytes(stored_image.data)
            except Exception as e:
                logger.error("Inference job %s failed to decode image", job_id)
                raise RuntimeError("Failed to decode inference image") from e

            report = generator_fn(
                tokenizer_config,
                vocabulary,
                image,
                stream_callback=None,
            )
            reports_by_filename[stored_image.filename] = report
            progress = (image_index / total_images) * 100.0
            job_manager.update_progress(job_id, progress)
            job_manager.update_result(
                job_id,
                {
                    "reports": dict(reports_by_filename),
                    "count": len(reports_by_filename),
                    "processed_images": image_index,
                    "total_images": total_images,
                },
            )
    finally:
        inference_image_store.remove_job(job_id)

    return {
        "reports": reports_by_filename,
        "count": len(reports_by_filename),
    }


###############################################################################
class InferenceEndpoint:
    """Endpoint for inference and report generation operations."""

    JOB_TYPE = "inference"

    def __init__(
        self,
        router: APIRouter,
        job_manager: JobManager,
    ) -> None:
        self.router = router
        self.job_manager = job_manager

    # -----------------------------------------------------------------------------
    def get_checkpoints(self) -> CheckpointsResponse:
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
    async def generate_reports(
        self,
        checkpoint: str = Form(...),
        generation_mode: str = Form(...),
        images: list[UploadFile] = File(...),
    ) -> JobStartResponse:
        if len(images) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No images provided",
            )

        if len(images) > MAX_INFERENCE_IMAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum {MAX_INFERENCE_IMAGES} images allowed",
            )

        allowed_modes = {"greedy_search", "beam_search"}
        if generation_mode not in allowed_modes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported generation mode: {generation_mode}",
            )

        checkpoint = checkpoint.strip()
        if not checkpoint:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Checkpoint name cannot be empty",
            )

        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint, "saved_model.keras")
        if not os.path.isfile(checkpoint_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint not found: {checkpoint}",
            )

        request_id = uuid.uuid4().hex[:12]
        total_bytes = 0
        stored_images: list[InferenceImage] = []
        try:
            for img in images:
                if img.filename is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Image upload missing filename",
                    )

                content_type = img.content_type or ""
                extension = os.path.splitext(img.filename)[1].lower()
                if (
                    content_type not in ALLOWED_IMAGE_TYPES
                    and extension not in ALLOWED_IMAGE_EXTENSIONS
                ):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported image type: {content_type or extension}",
                    )

                content = await img.read()
                if len(content) == 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Empty image payload: {img.filename}",
                    )

                total_bytes += len(content)
                if total_bytes > MAX_TOTAL_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            "Total image payload exceeds "
                            f"{MAX_TOTAL_IMAGE_BYTES // (1024 * 1024)} MB limit"
                        ),
                    )

                stored_images.append(
                    InferenceImage(
                        filename=img.filename,
                        content_type=content_type,
                        data=content,
                        size_bytes=len(content),
                    )
                )

            if not stored_images:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid images provided",
                )

            inference_image_store.store(request_id, stored_images)
            logger.info(
                "Staged inference images request %s with %d images (%d bytes)",
                request_id,
                len(stored_images),
                total_bytes,
            )

            # Start background job
            job_id = self.job_manager.start_job(
                job_type="inference",
                runner=run_inference_job,
                kwargs={
                    "checkpoint": checkpoint,
                    "generation_mode": generation_mode,
                    "request_id": request_id,
                },
            )

            inference_image_store.link_job(job_id, request_id)
            job_status = self.job_manager.get_job_status(job_id)
            if job_status is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to initialize inference job",
                )

            return JobStartResponse(
                job_id=job_id,
                job_type=job_status["job_type"],
                status=job_status["status"],
                message=f"Inference job started for {len(stored_images)} images",
            )

        except HTTPException:
            inference_image_store.remove_request(request_id)
            raise
        except Exception as e:
            inference_image_store.remove_request(request_id)
            logger.error(f"Error starting inference job: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            ) from e

    # -----------------------------------------------------------------------------
    async def get_inference_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return JobStatusResponse(**job_status)

    # -----------------------------------------------------------------------------
    async def cancel_inference_job(self, job_id: str) -> JobCancelResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )

        success = self.job_manager.cancel_job(job_id)
        if success:
            inference_image_store.remove_job(job_id)

        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )

    # -----------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all inference-related routes."""
        self.router.add_api_route(
            "/checkpoints",
            self.get_checkpoints,
            methods=["GET"],
            response_model=CheckpointsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/generate",
            self.generate_reports,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_inference_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_inference_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
router = APIRouter(prefix="/inference", tags=["inference"])
inference_endpoint = InferenceEndpoint(
    router=router,
    job_manager=job_manager,
)
inference_endpoint.add_routes()
