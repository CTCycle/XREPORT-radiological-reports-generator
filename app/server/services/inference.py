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
    ModelUpdateCheckResponse,
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
from server.models.inference.providers.huggingface import HuggingFaceProvider
from server.services.jobs import JobExecutionError, JobManager, get_job_manager
from server.repositories.serialization.inference import InferenceRepository
from server.configurations.startup import get_server_settings
from server.services.inference_catalog import InferenceModelCatalog
from server.services.inference_runtime import InferenceRuntimeCoordinator
from server.services.model_installation import (
    InstallationCancelled,
    InstallationError,
    ModelInstallationManager,
)


MAX_INFERENCE_IMAGES = 16
MAX_TOTAL_IMAGE_BYTES = 64 * 1024 * 1024

###############################################################################
def map_inference_failure(exc: Exception) -> JobExecutionError:
    if isinstance(exc, JobExecutionError):
        return exc

    message = str(exc).split("\n")[0][:300]
    lowered = message.lower()
    code = "inference_failed"
    phase = "inference"
    recoverable = True
    if "accepted model terms" in lowered or "valid local token" in lowered:
        code, phase = "access_required", "download"
    elif "download" in lowered or "hub" in lowered:
        code, phase = "download_failed", "download"
    elif "integrity" in lowered or "hash mismatch" in lowered or "size mismatch" in lowered:
        code, phase = "integrity_failed", "verify"
    elif "required runtime modules" in lowered or "requires the qwen" in lowered:
        code, phase = "runtime_dependency_missing", "loading"
    elif "cuda" in lowered or "out of memory" in lowered or "memory" in lowered:
        code, phase = "hardware_insufficient", "loading"
    elif "snapshot" in lowered or "checkpoint" in lowered:
        code, phase = "model_load_failed", "loading"
    elif "cancel" in lowered:
        code, phase, recoverable = "cancelled", "working", True
    return JobExecutionError(
        message,
        code=code,
        phase=phase,
        recoverable=recoverable,
    )

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
@lru_cache(maxsize=1)
def get_model_installation_manager() -> ModelInstallationManager:
    return ModelInstallationManager()

###############################################################################
@lru_cache(maxsize=1)
def get_inference_runtime() -> InferenceRuntimeCoordinator:
    return InferenceRuntimeCoordinator(
        huggingface_provider=get_huggingface_provider(),
        installation_manager=get_model_installation_manager(),
    )

###############################################################################
def report_installation_lifecycle(job_id: str, payload: dict[str, Any]) -> None:
    phase = str(payload.get("phase", "working"))
    weights = {
        "checking": (0.0, 5.0),
        "downloading": (5.0, 70.0),
        "verifying": (70.0, 82.0),
        "verified": (82.0, 86.0),
        "loading": (86.0, 94.0),
        "generating": (94.0, 99.0),
        "activating": (99.0, 100.0),
        "completed": (100.0, 100.0),
    }
    start, end = weights.get(phase, (0.0, 100.0))
    downloaded = payload.get("downloaded_bytes")
    total = payload.get("total_bytes")
    if phase == "downloading" and isinstance(downloaded, (int, float)) and isinstance(total, (int, float)) and total:
        progress = start + (end - start) * min(1.0, max(0.0, downloaded / total))
    else:
        progress = start
    get_job_manager().update_progress(job_id, progress)
    get_job_manager().update_result(job_id, {"lifecycle": payload})

###############################################################################
def report_inference_progress(
    job_id: str,
    image_index: int,
    total_images: int,
    reports: dict[str, str],
    inference_metadata: list[dict[str, Any]] | None = None,
    display_sections: dict[str, dict[str, str]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    progress = 94.0 + ((image_index / total_images) * 5.0)
    get_job_manager().update_progress(job_id, progress)
    get_job_manager().update_result(
        job_id,
        {
            "lifecycle": {
                "phase": "generating",
                "message": f"Generating report {image_index} of {total_images}",
                "processed_images": image_index,
                "total_images": total_images,
            },
        },
    )
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
    if inference_metadata is not None:
        get_job_manager().update_result(
            job_id,
            {"inference_metadata": inference_metadata},
        )
    if display_sections is not None:
        get_job_manager().update_result(
            job_id,
            {"display_sections": display_sections},
        )
    if provenance is not None:
        get_job_manager().update_result(job_id, {"provenance": provenance})

###############################################################################
def run_inference_job(
    model_ref: str,
    model_revision: str | None,
    model_manifest: dict[str, Any] | None,
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
    persisted_provenance: dict[str, Any] = {}
    try:
        report_progress = partial(report_inference_progress, job_id)
        execution = get_inference_runtime().generate(
            model_ref=model_ref,
            model_revision=model_revision,
            model_manifest=model_manifest,
            generation_profile=generation_profile,
            clinical_context=clinical_context,
            images=stored_images,
            should_stop=partial(get_job_manager().should_stop, job_id),
            report_progress=report_progress,
            report_lifecycle=partial(report_installation_lifecycle, job_id),
        )
        reports_by_filename = execution.reports
        display_sections = execution.display_sections
        provenance = execution.provenance
        reported_revision = provenance.get("model_revision")
        if isinstance(reported_revision, str):
            model_revision = reported_revision
    finally:
        inference_image_store.remove_job(job_id)

    if get_job_manager().should_stop(job_id) or not reports_by_filename:
        return {
            "reports": {},
            "reports_ordered": [],
            "report_filenames": [],
            "count": 0,
            "display_sections": {},
            "provenance": provenance,
        }

    reports_ordered = list(reports_by_filename.values())
    report_filenames = list(reports_by_filename)

    try:
        serializer = InferenceRepository()
        job_snapshot = get_job_manager().get_job_status(job_id) or {}
        job_result = job_snapshot.get("result") or {}
        persisted_provenance = dict(job_result.get("provenance", provenance))
        if "input_images" not in persisted_provenance:
            persisted_provenance["input_images"] = job_result.get("inference_metadata", [])
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
            generation_config={
                "profile": generation_profile,
                "inference_metadata": job_result.get("inference_metadata", []),
                "display_sections": job_result.get("display_sections", display_sections),
                "provenance": persisted_provenance,
            },
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
        "display_sections": display_sections,
        "provenance": persisted_provenance or provenance,
    }

###############################################################################
def run_model_maintenance_job(
    *,
    model_ref: str,
    manifest: dict[str, Any],
    action: str,
    revision: str,
    job_id: str,
) -> dict[str, Any]:
    manager = get_model_installation_manager()
    repository_id = model_ref.removeprefix("huggingface:")
    try:
        if action == "delete_local":
            deleted = get_inference_runtime().delete_local(repository_id, manifest)
            payload = {
                "phase": "completed",
                "message": "Local model files deleted; the public catalogue entry remains available.",
                "action": action,
                **deleted,
            }
            report_installation_lifecycle(job_id, payload)
            return {"lifecycle": payload, **deleted}
        candidate = manager.stage(
            manifest={**manifest, "revision": revision},
            revision=revision,
            should_stop=lambda: get_job_manager().should_stop(job_id),
            report_progress=partial(report_installation_lifecycle, job_id),
            operation_id=None,
            force_download=action == "reinstall",
        )
        return {
            "lifecycle": {
                "phase": "verified",
                "message": "Candidate model is verified and ready for validation through Generate",
                "revision": candidate.revision,
                "local_path": manager.relative_path(candidate.path),
                "action": action,
            },
            "candidate_revision": candidate.revision,
            "candidate_path": manager.relative_path(candidate.path),
        }
    except (InstallationCancelled, InstallationError) as exc:
        manager.record_error(
            repository_id,
            str(exc),
            state="failed",
            interrupted=(
                isinstance(exc, InstallationCancelled)
                or manager.is_resumable_error(str(exc))
            ),
        )
        raise

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
    def get_model_update(self, model_ref: str) -> ModelUpdateCheckResponse:
        selected = next((model for model in self.get_models().models if model.model_ref == model_ref), None)
        if selected is None or selected.provider != "huggingface":
            raise NotFoundError(detail=f"Model is not in the local inference catalog: {model_ref}")
        result = get_model_installation_manager().check_update(
            model_ref.removeprefix("huggingface:"),
        )
        return ModelUpdateCheckResponse(**result)

    # -------------------------------------------------------------------------
    def start_model_maintenance(
        self,
        *,
        model_ref: str,
        action: str,
        revision: str | None,
    ) -> JobStartResponse:
        catalog = self.get_models()
        selected = next((model for model in catalog.models if model.model_ref == model_ref), None)
        if selected is None:
            raise NotFoundError(detail=f"Model is not in the local inference catalog: {model_ref}")
        if selected.provider != "huggingface":
            raise UnsupportedOperationError(detail="Maintenance is only available for Hugging Face models")
        manifest = selected.model_dump(mode="json")
        manifest["repository_id"] = model_ref.removeprefix("huggingface:")
        configured_revision = str(manifest["model_revision"])
        if action not in {"download", "repair", "reinstall", "download_update", "delete_local"}:
            raise BadRequestError(detail=f"Unsupported model maintenance action: {action}")
        target_revision = revision or configured_revision
        if action != "delete_local" and (
            len(target_revision) != 40
            or any(character not in "0123456789abcdef" for character in target_revision)
        ):
            raise BadRequestError(detail="Maintenance revision must be a 40-character commit SHA")
        if action == "download_update" and revision is None:
            raise BadRequestError(detail="download_update requires the revision returned by check-update")
        if action == "delete_local":
            target_revision = configured_revision
        job_id = self.job_manager.start_job(
            job_type="model_maintenance",
            runner=run_model_maintenance_job,
            failure_mapper=map_inference_failure,
            kwargs={
                "model_ref": model_ref,
                "manifest": manifest,
                "action": action,
                "revision": target_revision,
            },
        )
        status = self.get_job_status_or_500(job_id, "Failed to initialize model maintenance job")
        return JobStartResponse(
            job_id=job_id,
            job_type=status["job_type"],
            status=status["status"],
            message=f"Model {action} started for {model_ref}",
            poll_interval=get_server_settings().jobs.polling_interval,
        )

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
        if selected_model.status not in {"ready", "not_installed", "unvalidated", "runtime_unavailable"}:
            raise ConflictError(
                detail=f"Model is not ready: {model_ref} ({selected_model.status})",
            )
        if selected_model.provider not in {
            "xreport",
            "huggingface",
        }:
            raise UnsupportedOperationError(
                detail=f"Generation is not implemented for provider: {selected_model.provider}",
            )
        if clinical_context and not selected_model.capabilities.clinical_context:
            raise BadRequestError(
                detail="Selected model does not support clinical context",
            )
        if len(images) > selected_model.max_current_images:
            raise BadRequestError(
                detail=(
                    "Selected model accepts at most "
                    f"{selected_model.max_current_images} current image(s)"
                ),
            )
        self.validate_inference_images(images)
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
            self.inference_image_store.store(request_id, images)

            # Start background job
            job_id = self.job_manager.start_job(
                job_type=self.JOB_TYPE,
                runner=run_inference_job,
                failure_mapper=map_inference_failure,
                kwargs={
                    "model_ref": model_ref,
                    "model_revision": selected_model.model_revision,
                    "model_manifest": {
                        **selected_model.model_dump(mode="json"),
                        "revision": selected_model.model_revision,
                    },
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
