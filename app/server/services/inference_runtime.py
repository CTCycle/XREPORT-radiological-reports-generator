from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

from server.domain.inference import (
    GenerationProfile,
    InferenceImage,
    ProviderGenerationResult,
)
from server.models.inference.providers.huggingface import HuggingFaceProvider
from server.models.inference.providers.xreport import XReportCheckpointProvider
from server.repositories.serialization.model import ModelSerializer
from server.services.model_installation import (
    InstallationCancelled,
    InstallationError,
    InstallationTarget,
    ModelInstallationManager,
)


StopCallback = Callable[[], bool]
LifecycleCallback = Callable[[dict[str, Any]], None]
ProgressCallback = Callable[..., None]


###############################################################################
class InferenceRuntimeCoordinator:
    """Serializes resident model use across the supported inference providers."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        huggingface_provider: HuggingFaceProvider,
        installation_manager: ModelInstallationManager,
    ) -> None:
        self.huggingface_provider = huggingface_provider
        self.installation_manager = installation_manager
        self.lock = threading.RLock()

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
            self.huggingface_provider.unload()
            try:
                model, _, model_metadata, _, _ = ModelSerializer().load_checkpoint(
                    checkpoint
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Checkpoint not found: {checkpoint}") from exc
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
            del model

        display_sections = {
            filename: {"raw_report": report}
            for filename, report in reports.items()
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
        report_lifecycle({
            "phase": "loading",
            "message": "Loading the verified local model snapshot",
            "revision": effective_revision,
            "local_path": self.installation_manager.relative_path(target.path),
        })

        with self.lock:
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
            report_lifecycle({
                "phase": "activating",
                "message": "Activating the verified revision after successful inference",
                "revision": effective_revision,
            })
            activated = self.installation_manager.activate(
                manifest=effective_manifest,
                target=target,
            )
            provenance["installation"]["state"] = "active"
            provenance["installation"]["local_path"] = activated.get(
                "active_relative_path",
                provenance["installation"]["local_path"],
            )
        report_lifecycle({
            "phase": "completed",
            "message": "Model loaded and report generated",
            "revision": effective_revision,
            "local_path": provenance["installation"]["local_path"],
        })
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
        target = (
            self.installation_manager.candidate_target(manifest)
            or self.installation_manager.active_target(manifest)
        )
        cloud_assessment: dict[str, Any] | None = None
        if target is not None:
            return target, cloud_assessment

        report_lifecycle({
            "phase": "checking",
            "message": "Checking whether a free cloud inference route is available",
            "model_ref": model_ref,
        })
        cloud_assessment = self.installation_manager.assess_cloud(repository_id)
        report_lifecycle({
            "phase": "checking",
            "message": cloud_assessment["reason"],
            "cloud_assessment": cloud_assessment,
        })
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
