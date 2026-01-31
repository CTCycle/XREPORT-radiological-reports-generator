from __future__ import annotations

import os
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
from XREPORT.server.utils.constants import RESOURCES_PATH, CHECKPOINT_PATH
from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.learning.inference import TextGenerator
from XREPORT.server.utils.jobs import JobManager, job_manager
from XREPORT.server.utils.repository.serializer import ModelSerializer


# Inference temp folder for uploaded images
INFERENCE_TEMP_PATH = os.path.join(RESOURCES_PATH, "inference_temp")


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
    tokenizers_info = generator.load_tokenizer_and_configuration()
    if tokenizers_info is None:
        raise RuntimeError("Failed to load tokenizer")

    tokenizer, tokenizer_config = tokenizers_info
    vocabulary = tokenizer.get_vocab()

    generator_fn = generator.generator_methods.get(generation_mode)
    if generator_fn is None:
        raise RuntimeError(f"Unknown generation mode: {generation_mode}")

    reports_by_filename: dict[str, str] = {}
    total_images = max(1, len(image_paths))

    for idx, path in enumerate(image_paths):
        if job_manager.should_stop(job_id):
            break

        report = generator_fn(
            tokenizer_config,
            vocabulary,
            path,
            stream_callback=None,
        )
        reports_by_filename[os.path.basename(path)] = report
        progress = ((idx + 1) / total_images) * 100.0
        job_manager.update_progress(job_id, progress)
        job_manager.update_result(
            job_id,
            {
                "reports": dict(reports_by_filename),
                "count": len(reports_by_filename),
                "processed_images": idx + 1,
                "total_images": total_images,
            },
        )

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
            job_id = self.job_manager.start_job(
                job_type="inference",
                runner=run_inference_job,
                kwargs={
                    "checkpoint": checkpoint,
                    "generation_mode": generation_mode,
                    "image_paths": image_paths,
                },
            )

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
