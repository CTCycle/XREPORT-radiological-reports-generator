from __future__ import annotations

import threading
import time
import uuid
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from server.services.errors import (
    BadRequestError,
    ConflictError,
    InternalServiceError,
    NotFoundError,
    PayloadTooLargeError,
    ServiceError,
    UnsupportedOperationError,
)

from server.domain.inference import (
    GenerationProfile,
    InferenceImage,
    InferenceModelsResponse,
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
from server.models.inference.providers.xreport import XReportCheckpointProvider
from server.models.inference.providers.ollama import OllamaProvider
from server.models.inference.providers.huggingface import HuggingFaceProvider
from server.services.jobs import JobManager, get_job_manager
from server.repositories.serialization.inference import InferenceRepository
from server.repositories.serialization.model import ModelSerializer
from server.configurations.startup import get_server_settings
from server.services.inference_catalog import InferenceModelCatalog


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
@lru_cache(maxsize=1)
def get_huggingface_provider() -> HuggingFaceProvider:
    return HuggingFaceProvider(get_server_settings().inference)

###############################################################################
def report_inference_progress(
    job_id: str,
    image_index: int,
    total_images: int,
    reports: dict[str, str],
) -> None:
    progress = (image_index / total_images) * 100.0
    get_job_manager().update_progress(job_id, progress)
    get_job_manager().update_result(
        job_id,
        {
            "reports": dict(reports),
            "reports_ordered": list(reports.values()),
            "report_filenames": list(reports),
            "count": len(reports),
            "processed_images": image_index,
            "total_images": total_images,
        },
    )

###############################################################################
def run_inference_job(
    model_ref: str,
    model_revision: str | None,
    generation_profile: GenerationProfile,
    clinical_context: str,
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

    started_at = time.perf_counter()
    try:
        report_progress = partial(report_inference_progress, job_id)

        if model_ref.startswith("xreport:"):
            generation_mode = {
                "deterministic": "greedy_search",
                "concise": "greedy_search",
                "detailed": "beam_search",
            }[generation_profile]
            checkpoint = model_ref.removeprefix("xreport:")
            try:
                model, _, model_metadata, _, _ = ModelSerializer().load_checkpoint(
                    checkpoint
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Checkpoint not found: {checkpoint}") from exc
            reports_by_filename = XReportCheckpointProvider().generate(
                model=model,
                model_metadata=model_metadata,
                generation_mode=generation_mode,
                images=stored_images,
                should_stop=lambda: get_job_manager().should_stop(job_id),
                report_progress=report_progress,
            )
        elif model_ref.startswith("ollama:"):
            reports_by_filename = OllamaProvider(
                get_server_settings().inference
            ).generate(
                model=model_ref.removeprefix("ollama:"),
                profile=generation_profile,
                clinical_context=clinical_context,
                images=stored_images,
                should_stop=lambda: get_job_manager().should_stop(job_id),
                report_progress=report_progress,
            )
        elif model_ref.startswith("huggingface:"):
            settings = get_server_settings().inference
            revision = settings.hf_medgemma_revision
            if revision is None:
                raise RuntimeError("MedGemma pinned revision is not configured")
            reports_by_filename = get_huggingface_provider().generate(
                repository_id=model_ref.removeprefix("huggingface:"),
                revision=revision,
                profile=generation_profile,
                clinical_context=clinical_context,
                images=stored_images,
                should_stop=lambda: get_job_manager().should_stop(job_id),
                report_progress=report_progress,
            )
        else:
            raise RuntimeError(f"Unsupported inference provider: {model_ref}")
    finally:
        inference_image_store.remove_job(job_id)

    reports_ordered = list(reports_by_filename.values())
    report_filenames = list(reports_by_filename)

    try:
        serializer = InferenceRepository()
        serializer.save_generated_reports(
            [
                {
                    "image": filename,
                    "report": report,
                }
                for filename, report in reports_by_filename.items()
            ],
            provider=model_ref.partition(":")[0],
            model_ref=model_ref,
            model_revision=model_revision,
            generation_profile=generation_profile,
            generation_config={"profile": generation_profile},
            clinical_context=clinical_context,
            request_id=request_id,
            status="succeeded",
            execution_time_seconds=time.perf_counter() - started_at,
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
            raise NotFoundError(
                detail=f"Job not found: {job_id}",
            )
        return job_status

    # -------------------------------------------------------------------------
    def get_job_status_or_500(self, job_id: str, detail: str) -> dict[str, Any]:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise InternalServiceError(
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
            raise BadRequestError(
                detail="No images provided",
            )

        if len(images) > MAX_INFERENCE_IMAGES:
            raise BadRequestError(
                detail=f"Maximum {MAX_INFERENCE_IMAGES} images allowed",
            )

        allowed_modes = {"greedy_search", "beam_search"}
        if generation_mode not in allowed_modes:
            raise BadRequestError(
                detail=f"Unsupported generation mode: {generation_mode}",
            )

        try:
            return XReportCheckpointProvider().validate_checkpoint(checkpoint)
        except FileNotFoundError as exc:
            raise NotFoundError(detail=str(exc)) from exc
        except ValueError as exc:
            raise BadRequestError(detail=str(exc)) from exc

    # -------------------------------------------------------------------------
    def validate_inference_images(
        self,
        images: list[InferenceImage],
    ) -> int:
        total_bytes = 0
        for image in images:
            filename = _sanitize_filename(image.filename.strip())
            if not filename:
                raise BadRequestError(
                    detail="Image upload missing filename",
                )
            content_type = image.content_type or ""
            extension = Path(filename).suffix.lower()
            if (
                content_type not in INFERENCE_IMAGE_CONTENT_TYPES
                and extension not in INFERENCE_IMAGE_EXTENSIONS
            ):
                raise BadRequestError(
                    detail=f"Unsupported image type: {content_type or extension}",
                )

            if len(image.data) == 0:
                raise BadRequestError(
                    detail=f"Empty image payload: {filename}",
                )

            total_bytes += len(image.data)
            if total_bytes > MAX_TOTAL_IMAGE_BYTES:
                raise PayloadTooLargeError(
                    detail=(
                        "Total image payload exceeds "
                        f"{MAX_TOTAL_IMAGE_BYTES // (1024 * 1024)} MB limit"
                    ),
                )

        return total_bytes

    # -------------------------------------------------------------------------
    def get_models(self) -> InferenceModelsResponse:
        return InferenceModelCatalog(get_server_settings().inference).list_models()

    # -------------------------------------------------------------------------
    def generate_reports(
        self,
        model_ref: str,
        generation_profile: GenerationProfile,
        clinical_context: str,
        images: list[InferenceImage],
    ) -> JobStartResponse:
        catalog = self.get_models()
        selected_model = next(
            (model for model in catalog.models if model.model_ref == model_ref),
            None,
        )
        if selected_model is None:
            raise NotFoundError(
                detail=f"Model is not in the local inference catalog: {model_ref}",
            )
        if selected_model.status != "ready":
            raise ConflictError(
                detail=f"Model is not ready: {model_ref} ({selected_model.status})",
            )
        if selected_model.provider not in {
            "xreport",
            "ollama",
            "huggingface",
        }:
            raise UnsupportedOperationError(
                detail=f"Generation is not implemented for provider: {selected_model.provider}",
            )
        if clinical_context and not selected_model.capabilities.clinical_context:
            raise BadRequestError(
                detail="Selected model does not support clinical context",
            )
        if selected_model.input_semantics == "single_image" and len(images) != 1:
            raise BadRequestError(
                detail="Selected model accepts exactly one image",
            )
        if selected_model.provider == "xreport":
            generation_mode = {
                "deterministic": "greedy_search",
                "concise": "greedy_search",
                "detailed": "beam_search",
            }[generation_profile]
            self.validate_generation_request(
                checkpoint=model_ref.removeprefix("xreport:"),
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
                    "model_ref": model_ref,
                    "model_revision": selected_model.model_revision,
                    "generation_profile": generation_profile,
                    "clinical_context": clinical_context,
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

        except ServiceError:
            self.inference_image_store.remove_request(request_id)
            raise
        except Exception as e:
            self.inference_image_store.remove_request(request_id)
            logger.error(f"Error starting inference job: {e}")
            raise InternalServiceError(
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
