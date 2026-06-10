from __future__ import annotations

import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from server.domain.inference import (
    CheckpointInfo,
    CheckpointsResponse,
    InferenceImage,
)
from server.domain.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from server.common.constants import (
    INFERENCE_IMAGE_CONTENT_TYPES,
    INFERENCE_IMAGE_EXTENSIONS,
)
from server.common.utils.logger import logger
from server.common.utils.security import resolve_checkpoint_path
from server.learning.inference import TextGenerator
from server.learning.training.dataloader import XRAYDataLoader
from server.services.jobs import JobManager, get_job_manager
from server.repositories.serialization.data import DataSerializer
from server.repositories.serialization.model import ModelSerializer
from server.configurations.startup import get_server_settings


MAX_INFERENCE_IMAGES = 16
MAX_TOTAL_IMAGE_BYTES = 64 * 1024 * 1024

###############################################################################
def _sanitize_filename(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name

###############################################################################
class InferenceImageStore:

    # -------------------------------------------------------------------------
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


###############################################################################
@lru_cache(maxsize=1)
def get_inference_image_store() -> InferenceImageStore:
    return InferenceImageStore()

###############################################################################
def run_inference_job(
    checkpoint: str,
    generation_mode: str,
    request_id: str,
    job_id: str,
) -> dict[str, Any]:
    """Blocking inference function that runs in background thread."""
    inference_image_store = get_inference_image_store()
    if get_job_manager().should_stop(job_id):
        inference_image_store.remove_job(job_id)
        return {}

    stored_images = inference_image_store.get(request_id)
    if stored_images is None or len(stored_images) == 0:
        logger.error("Inference job %s has no images to process", job_id)
        raise RuntimeError("No images available for inference job")

    # Load model checkpoint
    serializer = ModelSerializer()

    try:
        model, _, model_metadata, _, _ = serializer.load_checkpoint(
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
    reports_ordered: list[str] = []
    report_filenames: list[str] = []
    total_images = len(stored_images)
    dataloader = XRAYDataLoader(model_metadata, shuffle=False)

    try:
        for image_index, stored_image in enumerate(stored_images, start=1):
            if get_job_manager().should_stop(job_id):
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
            reports_ordered.append(report)
            report_filenames.append(stored_image.filename)
            progress = (image_index / total_images) * 100.0
            get_job_manager().update_progress(job_id, progress)
            get_job_manager().update_result(
                job_id,
                {
                    "reports": dict(reports_by_filename),
                    "reports_ordered": list(reports_ordered),
                    "report_filenames": list(report_filenames),
                    "count": len(reports_by_filename),
                    "processed_images": image_index,
                    "total_images": total_images,
                },
            )
    finally:
        inference_image_store.remove_job(job_id)

    try:
        serializer = DataSerializer()
        serializer.save_generated_reports(
            [
                {
                    "image": filename,
                    "report": report,
                    "checkpoint": checkpoint,
                }
                for filename, report in reports_by_filename.items()
            ],
            generation_mode=generation_mode,
            request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist generated reports for %s: %s", job_id, exc)

    return {
        "reports": reports_by_filename,
        "reports_ordered": reports_ordered,
        "report_filenames": report_filenames,
        "count": len(reports_by_filename),
    }

###############################################################################
class InferenceService:
    """Endpoint for inference and report generation operations."""

    JOB_TYPE = "inference"

    # -------------------------------------------------------------------------
    def __init__(
        self,
        job_manager: JobManager,
        inference_image_store: InferenceImageStore,
    ) -> None:
        self.job_manager = job_manager
        self.inference_image_store = inference_image_store

    # -------------------------------------------------------------------------
    def get_job_status_or_404(self, job_id: str) -> dict[str, Any]:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return job_status

    # -------------------------------------------------------------------------
    def get_job_status_or_500(self, job_id: str, detail: str) -> dict[str, Any]:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=detail,
            )
        return job_status

    # -------------------------------------------------------------------------
    def validate_generation_request(
        self,
        checkpoint: str,
        generation_mode: str,
        images: list[InferenceImage],
    ) -> str:
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

        try:
            checkpoint_dir = resolve_checkpoint_path(checkpoint)
            checkpoint = Path(checkpoint_dir).name
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )
        checkpoint_path = Path(checkpoint_dir) / "saved_model.keras"
        if not checkpoint_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint not found: {checkpoint}",
            )

        return checkpoint

    # -------------------------------------------------------------------------
    def validate_inference_images(
        self,
        images: list[InferenceImage],
    ) -> int:
        total_bytes = 0
        for image in images:
            filename = _sanitize_filename(image.filename.strip())
            if not filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Image upload missing filename",
                )
            content_type = image.content_type or ""
            extension = Path(filename).suffix.lower()
            if (
                content_type not in INFERENCE_IMAGE_CONTENT_TYPES
                and extension not in INFERENCE_IMAGE_EXTENSIONS
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported image type: {content_type or extension}",
                )

            if len(image.data) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Empty image payload: {filename}",
                )

            total_bytes += len(image.data)
            if total_bytes > MAX_TOTAL_IMAGE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        "Total image payload exceeds "
                        f"{MAX_TOTAL_IMAGE_BYTES // (1024 * 1024)} MB limit"
                    ),
                )

        return total_bytes

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def generate_reports(
        self,
        checkpoint: str,
        generation_mode: str,
        images: list[InferenceImage],
    ) -> JobStartResponse:
        checkpoint = self.validate_generation_request(
            checkpoint=checkpoint,
            generation_mode=generation_mode,
            images=images,
        )

        request_id = uuid.uuid4().hex[:12]
        try:
            self.validate_inference_images(images)
            self.inference_image_store.store(request_id, images)

            # Start background job
            job_id = self.job_manager.start_job(
                job_type=self.JOB_TYPE,
                runner=run_inference_job,
                kwargs={
                    "checkpoint": checkpoint,
                    "generation_mode": generation_mode,
                    "request_id": request_id,
                },
            )

            self.inference_image_store.link_job(job_id, request_id)
            job_status = self.get_job_status_or_500(
                job_id=job_id,
                detail="Failed to initialize inference job",
            )

            return JobStartResponse(
                job_id=job_id,
                job_type=job_status["job_type"],
                status=job_status["status"],
                message=f"Inference job started for {len(images)} images",
                poll_interval=get_server_settings().jobs.polling_interval,
            )

        except HTTPException:
            self.inference_image_store.remove_request(request_id)
            raise
        except Exception as e:
            self.inference_image_store.remove_request(request_id)
            logger.error(f"Error starting inference job: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            ) from e

    # -------------------------------------------------------------------------
    def get_inference_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.get_job_status_or_404(job_id)
        return JobStatusResponse(**job_status)

    # -------------------------------------------------------------------------
    def cancel_inference_job(self, job_id: str) -> JobCancelResponse:
        self.get_job_status_or_404(job_id)

        success = self.job_manager.cancel_job(job_id)
        if success:
            self.inference_image_store.remove_job(job_id)

        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )

###############################################################################
@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService(
        job_manager=get_job_manager(),
        inference_image_store=get_inference_image_store(),
    )


