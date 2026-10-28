from __future__ import annotations

from collections.abc import Callable
import gc
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any

from xreport_inference.domain.inference import (
    GenerationProfile,
    InferenceImage,
    ProviderGenerationResult,
)
from xreport_inference.services.model_installation import (
    InstallationCancelled,
    InstallationError,
    InstallationTarget,
    ModelInstallationManager,
)
from xreport_inference.services.model_storage import ModelStorageLifecycle
from xreport_shared.repositories.checkpoints import CheckpointRepository
from xreport_shared.common.utils.logger import logger

if TYPE_CHECKING:
    from xreport_inference.models.inference.providers.huggingface import HuggingFaceProvider


StopCallback = Callable[[], bool]
LifecycleCallback = Callable[[dict[str, Any]], None]
ProgressCallback = Callable[..., None]
IdleTimeoutProvider = Callable[[], int]


###############################################################################
def release_framework_memory(*, keras_session: bool = False) -> None:
    """Collect dropped model graphs and return cached CUDA blocks to the driver."""
    if keras_session and "keras" in sys.modules:
        import keras

        keras.backend.clear_session(free_memory=True)
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


###############################################################################
class InferenceRuntimeCoordinator:
    """Serializes resident model use across the supported inference providers.

    At most one model is resident. It stays loaded between runs for reuse and
    is released after the configured idle timeout or on explicit request. A
    release only happens while the residency lock is free, so it never
    interrupts an active run.
    """

    idle_check_interval = 5.0

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        huggingface_provider: HuggingFaceProvider,
        installation_manager: ModelInstallationManager,
        storage_lifecycle: ModelStorageLifecycle | None = None,
        checkpoint_repository: CheckpointRepository | None = None,
        idle_timeout_provider: IdleTimeoutProvider | None = None,
    ) -> None:
        self.huggingface_provider = huggingface_provider
        self.installation_manager = installation_manager
        self.storage_lifecycle = storage_lifecycle or ModelStorageLifecycle(
            installation_manager
        )
        self.checkpoint_repository = checkpoint_repository or CheckpointRepository()
        self.idle_timeout_provider = idle_timeout_provider
        self.lock = threading.RLock()
        self.last_used = time.monotonic()
        self._xreport_ref: str | None = None
        self._xreport_key: tuple[str, int] | None = None
        self._xreport_model: Any = None
        self._xreport_metadata: dict[str, Any] | None = None
        self._watcher_lock = threading.Lock()
        self._idle_watcher: threading.Thread | None = None

    # -------------------------------------------------------------------------
    @property
    def resident_model(self) -> str | None:
        if self._xreport_ref is not None:
            return self._xreport_ref
        repository_id = self.huggingface_provider.loaded_repository
        return f"huggingface:{repository_id}" if repository_id else None

    # -------------------------------------------------------------------------
    def generate(
        self,
        *,
        model_ref: str,
        model_revision: str | None,
        model_manifest: dict[str, Any] | None,
        generation_profile: GenerationProfile,
        clinical_context: str,
        images: list[InferenceImage],
        should_stop: StopCallback,
        report_progress: ProgressCallback,
        report_lifecycle: LifecycleCallback,
    ) -> ProviderGenerationResult:
        try:
            return self._dispatch(
                model_ref=model_ref,
                model_revision=model_revision,
                model_manifest=model_manifest,
                generation_profile=generation_profile,
                clinical_context=clinical_context,
                images=images,
                should_stop=should_stop,
                report_progress=report_progress,
                report_lifecycle=report_lifecycle,
            )
        finally:
            self.last_used = time.monotonic()
            self._ensure_idle_watcher()

    # -------------------------------------------------------------------------
    def _dispatch(
        self,
        *,
        model_ref: str,
        model_revision: str | None,
        model_manifest: dict[str, Any] | None,
        generation_profile: GenerationProfile,
        clinical_context: str,
        images: list[InferenceImage],
        should_stop: StopCallback,
        report_progress: ProgressCallback,
        report_lifecycle: LifecycleCallback,
    ) -> ProviderGenerationResult:
        if model_ref.startswith("xreport:"):
            return self._generate_xreport(
                model_ref=model_ref,
                model_revision=model_revision,
                generation_profile=generation_profile,
                clinical_context=clinical_context,
                images=images,
                should_stop=should_stop,
                report_progress=report_progress,
            )
        if model_ref.startswith("huggingface:"):
            return self._generate_huggingface(
                model_ref=model_ref,
                model_revision=model_revision,
                model_manifest=model_manifest,
                generation_profile=generation_profile,
                clinical_context=clinical_context,
                images=images,
                should_stop=should_stop,
                report_progress=report_progress,
                report_lifecycle=report_lifecycle,
            )
        raise RuntimeError(f"Unsupported inference provider: {model_ref}")

    # -------------------------------------------------------------------------
    def delete_local(
        self,
        repository_id: str,
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Unload an idle public model and delete only its local snapshot."""
        acquired = self.lock.acquire(blocking=False)
        if not acquired:
            raise InstallationError(
                "The model is currently in use; wait for inference to finish before deleting local files."
            )
        try:
            self._release_resident()
            return self.storage_lifecycle.delete_local(
                repository_id,
                processor_repository_id=(manifest or {}).get("processor_repository_id"),
                processor_revision=(manifest or {}).get("processor_revision"),
            )
        finally:
            self.lock.release()

    # -------------------------------------------------------------------------
    def release(self, *, blocking: bool = False) -> bool:
        """Unload the resident model. Returns False while a run holds the lock."""
        if not self.lock.acquire(blocking=blocking):
            return False
        try:
            if self.resident_model is None:
                # Still return cached blocks left by earlier or failed runs.
                release_framework_memory()
            else:
                self._release_resident()
        finally:
            self.lock.release()
        return True

    # -------------------------------------------------------------------------
    def _release_resident(self) -> None:
        self.huggingface_provider.unload()
        self._drop_xreport()

    # -------------------------------------------------------------------------
    def _drop_xreport(self) -> None:
        if self._xreport_key is None:
            return
        self._xreport_ref = None
        self._xreport_key = None
        self._xreport_model = None
        self._xreport_metadata = None
        release_framework_memory(keras_session=True)

    # -------------------------------------------------------------------------
    def _ensure_idle_watcher(self) -> None:
        if self.idle_timeout_provider is None or self.resident_model is None:
            return
        with self._watcher_lock:
            if self._idle_watcher is not None:
                return
            self._idle_watcher = threading.Thread(
                target=self._watch_idle,
                name="inference-idle-release",
                daemon=True,
            )
            self._idle_watcher.start()

    # -------------------------------------------------------------------------
    def _watch_idle(self) -> None:
        # Exits once nothing is resident; the next run starts a new watcher.
        while True:
            with self._watcher_lock:
                if self.resident_model is None:
                    self._idle_watcher = None
                    return
            time.sleep(self.idle_check_interval)
            try:
                timeout = int(self.idle_timeout_provider())  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Idle model timeout unavailable: %s", exc)
                continue
            if timeout <= 0 or time.monotonic() - self.last_used < timeout:
                continue
            model_ref = self.resident_model
            if model_ref is not None and self.release(blocking=False):
                logger.info(
                    "Released idle inference model %s after %s s", model_ref, timeout
                )

    # -------------------------------------------------------------------------
    def _load_xreport(
        self, model_ref: str, checkpoint_path: str
    ) -> tuple[Any, dict[str, Any]]:
        """Reuse the resident checkpoint model, loading it only when it changed."""
        from xreport_shared.repositories.serialization.model import ModelSerializer

        weights = Path(checkpoint_path) / "saved_model.keras"
        key = (
            str(checkpoint_path),
            weights.stat().st_mtime_ns if weights.exists() else 0,
        )
        if self._xreport_key == key and self._xreport_model is not None:
            return self._xreport_model, self._xreport_metadata or {}
        self._release_resident()
        try:
            model, _, model_metadata, _, _ = ModelSerializer().load_checkpoint(
                checkpoint_path
            )
        except Exception as exc:  # noqa: BLE001
            # Clear the failed loader frames so partial weights can be collected.
            traceback.clear_frames(exc.__traceback__)
            release_framework_memory(keras_session=True)
            raise RuntimeError(
                f"Checkpoint not found: {model_ref.removeprefix('xreport:')}"
            ) from exc
        self._xreport_ref = model_ref
        self._xreport_key = key
        self._xreport_model = model
        self._xreport_metadata = model_metadata
        return model, model_metadata

    # -------------------------------------------------------------------------
    @staticmethod
    def _xreport_runtime_metadata(
        model: Any,
        requested_device: str,
    ) -> dict[str, Any]:
        """Capture the device actually holding the loaded Keras/Torch weights."""
        import torch

        resolved_devices: list[str] = []
        for variable in getattr(model, "variables", ()):
            value = getattr(variable, "value", None)
            device = getattr(value, "device", None)
            if device is None:
                device = getattr(variable, "device", None)
            if device is None:
                continue
            label = str(device)
            if label not in resolved_devices:
                resolved_devices.append(label)

        return {
            "requested_device": requested_device,
            "resolved_device": resolved_devices[0] if resolved_devices else None,
            "resolved_devices": resolved_devices,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_used": any(
                device == "cuda" or device.startswith("cuda:")
                for device in resolved_devices
            ),
        }

    # -------------------------------------------------------------------------
    def _generate_xreport(
        self,
        *,
        model_ref: str,
        model_revision: str | None,
        generation_profile: GenerationProfile,
        clinical_context: str,
        images: list[InferenceImage],
        should_stop: StopCallback,
        report_progress: ProgressCallback,
    ) -> ProviderGenerationResult:
        checkpoint = model_ref.removeprefix("xreport:")
        checkpoint_record = self.checkpoint_repository.get_checkpoint(checkpoint)
        if checkpoint_record is None:
            raise RuntimeError(f"Checkpoint is not registered: {checkpoint}")
        if not checkpoint_record.artifact_complete:
            raise RuntimeError(
                f"Checkpoint artifact is missing or incomplete: {checkpoint}"
            )
        provenance: dict[str, Any] = {
            "provider": "xreport",
            "model_ref": model_ref,
            "model_revision": model_revision,
            "generation_profile": generation_profile,
            "clinical_context": clinical_context,
            "research_only": True,
            "adapter": "xreport_beit",
            "model_loader": "keras_checkpoint",
            "processor_loader": "fixed_224",
        }
        if should_stop():
            return ProviderGenerationResult({}, {}, [], provenance)

        with self.lock:
            if should_stop():
                return ProviderGenerationResult({}, {}, [], provenance)
            from xreport_inference.models.inference.providers.xreport import (
                XReportCheckpointProvider,
            )

            model, model_metadata = self._load_xreport(
                model_ref, checkpoint_record.path
            )
            provenance["runtime"] = self._xreport_runtime_metadata(
                model,
                self.huggingface_provider.settings.device,
            )
            generation_mode = {
                "deterministic": "greedy_search",
                "concise": "greedy_search",
                "detailed": "beam_search",
            }[generation_profile]
            reports = XReportCheckpointProvider().generate(
                model=model,
                model_metadata=model_metadata,
                generation_mode=generation_mode,
                images=images,
                should_stop=should_stop,
                report_progress=report_progress,
            )

        display_sections = {
            filename: {"raw_report": report} for filename, report in reports.items()
        }
        return ProviderGenerationResult(
            reports=reports,
            display_sections=display_sections,
            metadata=[],
            provenance=provenance,
        )

    # -------------------------------------------------------------------------
    def _generate_huggingface(
        self,
        *,
        model_ref: str,
        model_revision: str | None,
        model_manifest: dict[str, Any] | None,
        generation_profile: GenerationProfile,
        clinical_context: str,
        images: list[InferenceImage],
        should_stop: StopCallback,
        report_progress: ProgressCallback,
        report_lifecycle: LifecycleCallback,
    ) -> ProviderGenerationResult:
        if model_manifest is None:
            raise RuntimeError("Hugging Face model manifest is missing")
        repository_id = model_ref.removeprefix("huggingface:")
        effective_manifest = dict(model_manifest)
        effective_manifest["repository_id"] = repository_id
        self._require_complete_manifest(effective_manifest)
        with self.lock:
            # Hold the residency lock across resolution, loading, generation,
            # and candidate activation so delete_local cannot race an active
            # inference job after it has staged a snapshot.
            target, cloud_assessment = self._resolve_target(
                repository_id=repository_id,
                manifest=effective_manifest,
                should_stop=should_stop,
                report_lifecycle=report_lifecycle,
                model_ref=model_ref,
            )
            effective_manifest["revision"] = target.revision
            effective_manifest["local_snapshot_path"] = str(target.path)
            effective_revision = target.revision
            report_lifecycle(
                {
                    "phase": "loading",
                    "message": "Loading the verified local model snapshot",
                    "revision": effective_revision,
                    "local_path": self.installation_manager.relative_path(target.path),
                }
            )
            self._drop_xreport()
            generation = self.huggingface_provider.generate(
                repository_id=repository_id,
                manifest=effective_manifest,
                profile=generation_profile,
                clinical_context=clinical_context,
                images=images,
                should_stop=should_stop,
                report_progress=report_progress,
            )
            self.installation_manager.record_success(repository_id, inference=True)
            provenance = dict(generation.provenance)
            provenance["input_images"] = generation.metadata
            provenance["installation"] = {
                "state": "candidate" if target.candidate else "active",
                "revision": effective_revision,
                "local_path": self.installation_manager.relative_path(target.path),
                "cloud_assessment": cloud_assessment,
            }
            if target.candidate and generation.reports:
                report_lifecycle(
                    {
                        "phase": "activating",
                        "message": "Activating the verified revision after successful inference",
                        "revision": effective_revision,
                    }
                )
                activated = self.installation_manager.activate(
                    manifest=effective_manifest,
                    target=target,
                )
                provenance["installation"]["state"] = "active"
                provenance["installation"]["local_path"] = activated.get(
                    "active_relative_path",
                    provenance["installation"]["local_path"],
                )
            report_lifecycle(
                {
                    "phase": "completed",
                    "message": "Model loaded and report generated",
                    "revision": effective_revision,
                    "local_path": provenance["installation"]["local_path"],
                }
            )
            return ProviderGenerationResult(
                reports=generation.reports,
                display_sections=generation.display_sections,
                metadata=generation.metadata,
                provenance=provenance,
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _require_complete_manifest(manifest: dict[str, Any]) -> None:
        required_fields = {
            "revision",
            "required_files",
            "weight_file_sets",
            "model_loader",
            "processor_loader",
            "adapter",
            "max_current_images",
            "preferred_dtype",
            "trust_remote_code",
        }
        missing = sorted(field for field in required_fields if field not in manifest)
        if missing:
            raise RuntimeError(
                "Hugging Face model manifest is incomplete: " + ", ".join(missing)
            )
        if not manifest["required_files"] or not manifest["weight_file_sets"]:
            raise RuntimeError(
                "Hugging Face model manifest must define required files and weight alternatives"
            )

    # -------------------------------------------------------------------------
    def _resolve_target(
        self,
        *,
        repository_id: str,
        manifest: dict[str, Any],
        should_stop: StopCallback,
        report_lifecycle: LifecycleCallback,
        model_ref: str,
    ) -> tuple[InstallationTarget, dict[str, Any] | None]:
        target = self.installation_manager.candidate_target(
            manifest
        ) or self.installation_manager.active_target(manifest)
        cloud_assessment: dict[str, Any] | None = None
        if target is not None:
            return target, cloud_assessment

        report_lifecycle(
            {
                "phase": "checking",
                "message": "Checking whether a free cloud inference route is available",
                "model_ref": model_ref,
            }
        )
        cloud_assessment = self.installation_manager.assess_cloud(repository_id)
        report_lifecycle(
            {
                "phase": "checking",
                "message": cloud_assessment["reason"],
                "cloud_assessment": cloud_assessment,
            }
        )
        revision = str(manifest["revision"])
        try:
            target = self.installation_manager.stage(
                manifest=manifest,
                revision=revision,
                should_stop=should_stop,
                report_progress=report_lifecycle,
            )
        except (InstallationCancelled, InstallationError) as exc:
            self.installation_manager.record_error(
                repository_id,
                str(exc),
                state="failed",
                interrupted=(
                    isinstance(exc, InstallationCancelled)
                    or self.installation_manager.is_resumable_error(str(exc))
                ),
            )
            raise
        return target, cloud_assessment
