from __future__ import annotations

import importlib.util
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

from server.common.path import (
    DATA_ROOT,
    PACKAGED_MODE,
    ROOT_DIR,
    SETTINGS_DIR,
)
from server.common.inference_manifest import validate_manifest
from server.configurations import InferenceSettings
from server.domain.inference import (
    InferenceManifest,
    InferenceManifestEntry,
    InferenceModelsResponse,
    ModelAvailability,
    ModelCapabilities,
    ProviderAvailability,
)
from server.services.model_installation import ModelInstallationManager
from server.repositories.checkpoints import CheckpointRepository


CATALOG_PATH = SETTINGS_DIR / "inference_models.json"
CXRMATE_ED_PROFILE_CONTRACT_VERSION = 1
VALIDATION_RECEIPTS_DIR = (
    DATA_ROOT / "validation_receipts"
    if PACKAGED_MODE
    else ROOT_DIR / "assets" / "QA" / "inference_validation"
)

###############################################################################
def validation_contract_hash(entry: InferenceManifestEntry) -> str:
    contract = {
        "repository_id": entry.repository_id,
        "revision": entry.revision,
        "anatomy_coverage": entry.anatomy_coverage,
        "hardware_demand": entry.hardware_demand,
        "model_loader": entry.model_loader,
        "processor_loader": entry.processor_loader,
        "adapter": entry.adapter,
        "prompt_profile": entry.prompt_profile,
        "output_sections": entry.output_sections,
        "input_semantics": entry.input_semantics,
        "supports_clinical_context": entry.supports_clinical_context,
        "supports_prior_images": entry.supports_prior_images,
        "capabilities": entry.capabilities.model_dump(mode="json"),
        "preferred_dtype": entry.preferred_dtype,
        "quantization": entry.quantization,
        "max_current_images": entry.max_current_images,
        "processor_repository_id": entry.processor_repository_id,
        "processor_revision": entry.processor_revision,
        "processor_files": entry.processor_files,
        "processor_target_prefix": entry.processor_target_prefix,
        "required_files": entry.required_files,
        "weight_file_sets": entry.weight_file_sets,
        "trust_remote_code": entry.trust_remote_code,
        "remote_code_approved": entry.remote_code_approved,
        "generation_profile_contract_version": (
            CXRMATE_ED_PROFILE_CONTRACT_VERSION
            if entry.adapter == "cxrmate_ed"
            else None
        ),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

###############################################################################
class InferenceModelCatalog:
    """Lists strict embedded model metadata and discovered XREPORT checkpoints."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        settings: InferenceSettings,
        installation_manager: ModelInstallationManager | None = None,
        checkpoint_repository: CheckpointRepository | None = None,
    ) -> None:
        self.settings = settings
        self.installation_manager = (
            installation_manager
            if installation_manager is not None
            else ModelInstallationManager()
        )
        self.checkpoint_repository = checkpoint_repository or CheckpointRepository()

    # -------------------------------------------------------------------------
    def list_models(self) -> InferenceModelsResponse:
        models = self._configured_models()
        xreport_models = self._xreport_models()
        models.extend(xreport_models)
        return InferenceModelsResponse(
            models=models,
            providers={
                "huggingface": self._huggingface_provider_status(models),
                "xreport": self._xreport_provider_status(xreport_models),
            },
        )

    # -------------------------------------------------------------------------
    def _configured_models(self) -> list[ModelAvailability]:
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        manifest = InferenceManifest.model_validate(payload)
        return [self._configured_model(entry) for entry in manifest.models]

    # -------------------------------------------------------------------------
    def _configured_model(
        self,
        entry: InferenceManifestEntry,
    ) -> ModelAvailability:
        status = "not_installed"
        status_message: str | None = None
        installation = self.installation_manager
        has_validation_evidence = self._has_validation_evidence(entry)
        inspected = installation.inspect(entry.model_dump(mode="json"))
        metadata = inspected["metadata"]
        installation_state = str(inspected["state"])
        local_path = inspected["active_path"]
        active_revision = inspected["active_revision"]
        candidate_revision = inspected["candidate_revision"]
        if not self.settings.hf_local_only:
            status = "disabled"
            status_message = "HF_LOCAL_ONLY must remain enabled for local inference."
        elif not entry.enabled:
            status = "disabled"
            status_message = entry.validation_message or "This model is disabled by policy."
        else:
            try:
                status_message = self._runtime_constraint_message(entry)
                if status_message is not None:
                    status = "incompatible"
                else:
                    validate_manifest(
                        entry.repository_id,
                        entry.model_dump(mode="json"),
                    )
                if status != "incompatible" and installation_state == "active" and local_path:
                    status = "ready"
                    status_message = None
                elif status != "incompatible" and installation_state == "staged":
                    status = "unvalidated"
                    status_message = "A verified candidate is installed and must pass real inference before activation."
                elif status != "incompatible" and installation_state == "downloading":
                    status = "downloading"
                    status_message = "A local model download is in progress."
                elif status != "incompatible" and installation_state in {"corrupt", "failed"}:
                    status = "runtime_unavailable"
                    status_message = str(metadata.get("last_error") or "The local installation needs repair.")
                elif status != "incompatible":
                    status = "not_installed"
                    status_message = (
                        entry.validation_message
                        if entry.gated
                        else "The model will be downloaded into the project-local resources directory on first Generate."
                    )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                status = "incompatible"
                status_message = str(exc)

        return ModelAvailability.model_validate({
            "model_ref": entry.model_ref,
            "provider": entry.provider,
            "origin": "public",
            "display_name": entry.display_name,
            "description": entry.description,
            "status": status,
            "status_message": status_message,
            "enabled": entry.enabled,
            "validation_status": entry.validation_status,
            "validation_message": entry.validation_message,
            "validation_receipt_status": "passed" if has_validation_evidence else "missing",
            "validation_receipt_message": (
                None
                if has_validation_evidence
                else "No valid real-inference validation receipt is recorded for this revision."
            ),
            "category": entry.category,
            "recommended": entry.recommended,
            "research_only": entry.research_only,
            "gated": entry.gated,
            "access_policy": entry.access_policy,
            "access_url": entry.access_url,
            "anatomy_coverage": entry.anatomy_coverage,
            "coverage_note": entry.coverage_note,
            "hardware_demand": entry.hardware_demand,
            "parameter_label": entry.parameter_label,
            "parameter_size": entry.parameter_size,
            "download_size_bytes": entry.download_size_bytes or entry.local_size_bytes,
            "local_size_bytes": entry.local_size_bytes,
            "input_semantics": entry.input_semantics,
            "capabilities": entry.capabilities,
            "model_revision": active_revision or entry.revision,
            "model_loader": entry.model_loader,
            "processor_loader": entry.processor_loader,
            "adapter": entry.adapter,
            "trust_remote_code": entry.trust_remote_code,
            "remote_code_approved": entry.remote_code_approved,
            "output_sections": entry.output_sections,
            "max_current_images": entry.max_current_images,
            "supports_prior_images": entry.supports_prior_images,
            "supports_clinical_context": entry.supports_clinical_context,
            "preferred_dtype": entry.preferred_dtype,
            "quantization": entry.quantization,
            "prompt_profile": entry.prompt_profile,
            "license": entry.license,
            "resource_policy": entry.resource_policy,
            "runtime_constraints": entry.runtime_constraints,
            "processor_repository_id": entry.processor_repository_id,
            "processor_revision": entry.processor_revision,
            "processor_files": entry.processor_files,
            "processor_target_prefix": entry.processor_target_prefix,
            "required_files": entry.required_files,
            "weight_file_sets": entry.weight_file_sets,
            "installation_state": installation_state if installation_state in {"not_installed", "staged", "active", "corrupt", "failed", "downloading"} else "not_installed",
            "local_path": ModelInstallationManager._relative(local_path) if local_path else None,
            "active_revision": active_revision,
            "candidate_revision": candidate_revision,
            "integrity_status": str(inspected["integrity"]),
            "cloud_assessment": metadata.get("cloud_assessment"),
            "update_available": bool((metadata.get("update_check") or {}).get("update_available")),
            "available_actions": (
                ["check_updates", "reinstall", "download_update", "delete_local"]
                if installation_state == "active"
                else ["repair", "delete_local"] if installation_state in {"staged", "corrupt", "failed", "downloading"}
                else ["download"] if status not in {"disabled", "incompatible"} else []
            ),
        })

    # -------------------------------------------------------------------------
    @staticmethod
    def _version_tuple(value: str) -> tuple[int, ...]:
        parts: list[int] = []
        for piece in value.split("."):
            digits = "".join(character for character in piece if character.isdigit())
            if not digits:
                break
            parts.append(int(digits))
        return tuple(parts)

    # -------------------------------------------------------------------------
    @classmethod
    def _runtime_constraint_message(
        cls,
        entry: InferenceManifestEntry,
    ) -> str | None:
        try:
            installed_version = package_version("transformers")
        except PackageNotFoundError:
            installed_version = "unavailable"
        current = cls._version_tuple(installed_version)
        constraints = entry.runtime_constraints
        if constraints.min_transformers and current < cls._version_tuple(constraints.min_transformers):
            return (
                f"Requires Transformers >= {constraints.min_transformers}; "
                f"the installed version is {installed_version}."
            )
        if constraints.max_transformers_exclusive and current >= cls._version_tuple(
            constraints.max_transformers_exclusive
        ):
            return (
                f"Requires Transformers < {constraints.max_transformers_exclusive}; "
                f"the installed version is {installed_version}."
            )
        missing = [
            module
            for module in constraints.required_modules
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            return "Required runtime modules are unavailable: " + ", ".join(missing)
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _status_priority(status: str) -> int:
        return {
            "ready": 0,
            "unvalidated": 1,
            "not_installed": 2,
            "downloading": 3,
            "gated": 4,
            "runtime_unavailable": 5,
            "incompatible": 6,
            "disabled": 7,
        }[status]

    # -------------------------------------------------------------------------
    @staticmethod
    def _validation_receipt_path(entry: InferenceManifestEntry) -> Path:
        slug = entry.repository_id.replace("/", "__")
        return VALIDATION_RECEIPTS_DIR / f"{slug}-{entry.revision}.json"

    # -------------------------------------------------------------------------
    @classmethod
    def _has_validation_evidence(cls, entry: InferenceManifestEntry) -> bool:
        receipt_path = cls._validation_receipt_path(entry)
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            isinstance(receipt, dict)
            and receipt.get("status") == "passed"
            and receipt.get("real_inference") is True
            and receipt.get("model_ref") == entry.model_ref
            and receipt.get("revision") == entry.revision
            and receipt.get("contract_hash") == validation_contract_hash(entry)
            and isinstance(receipt.get("reports"), dict)
            and bool(receipt["reports"])
            and all(
                isinstance(filename, str)
                and isinstance(report, str)
                and bool(report.strip())
                for filename, report in receipt["reports"].items()
            )
            and isinstance(receipt.get("display_sections"), dict)
            and set(receipt["display_sections"]) == set(receipt["reports"])
            and all(
                isinstance(sections, dict)
                and set(sections) == set(entry.output_sections)
                and all(isinstance(value, str) and bool(value.strip()) for value in sections.values())
                for sections in receipt["display_sections"].values()
            )
            and (
                "raw_report" not in entry.output_sections
                or all(
                    receipt["display_sections"][filename].get("raw_report") == report
                    for filename, report in receipt["reports"].items()
                )
            )
        )

    # -------------------------------------------------------------------------
    def _huggingface_provider_status(
        self,
        models: list[ModelAvailability],
    ) -> ProviderAvailability:
        hf_models = [model for model in models if model.provider == "huggingface"]
        if not self.settings.hf_local_only:
            return ProviderAvailability(
                status="disabled",
                message="HF_LOCAL_ONLY must remain enabled for local inference.",
            )
        if not hf_models:
            return ProviderAvailability(
                status="not_installed",
                message="No embedded Hugging Face model is configured.",
            )
        if any(model.status == "ready" for model in hf_models):
            return ProviderAvailability(status="ready")
        selected = min(hf_models, key=lambda model: self._status_priority(model.status))
        return ProviderAvailability(status=selected.status, message=selected.status_message)

    # -------------------------------------------------------------------------
    @staticmethod
    def _xreport_provider_status(
        models: list[ModelAvailability],
    ) -> ProviderAvailability:
        if any(model.status == "ready" for model in models):
            return ProviderAvailability(status="ready")
        if any(model.status == "runtime_unavailable" for model in models):
            return ProviderAvailability(
                status="runtime_unavailable",
                message="A registered XREPORT checkpoint artifact is unavailable.",
            )
        return ProviderAvailability(
            status="not_installed",
            message="No XREPORT checkpoints are registered.",
        )

    # -------------------------------------------------------------------------
    def _xreport_models(self) -> list[ModelAvailability]:
        checkpoints = self.checkpoint_repository.list_checkpoints()
        return [
            ModelAvailability(
                model_ref=f"xreport:{checkpoint.name}",
                provider="xreport",
                origin="custom",
                display_name=checkpoint.name,
                description=(
                    "Local XREPORT checkpoint using the fixed BEiT 224x224x3 "
                    "image encoder"
                ),
                status="ready" if checkpoint.artifact_complete else "runtime_unavailable",
                status_message=(
                    None
                    if checkpoint.artifact_complete
                    else "The registered checkpoint artifact is missing or incomplete."
                ),
                enabled=True,
                validation_status="passed" if checkpoint.artifact_complete else "blocked",
                validation_message=(
                    None
                    if checkpoint.artifact_complete
                    else "Restore the registered checkpoint artifact before using it."
                ),
                validation_receipt_status="passed" if checkpoint.artifact_complete else "missing",
                validation_receipt_message=(
                    None
                    if checkpoint.artifact_complete
                    else "The registered checkpoint artifact is not complete."
                ),
                category="xreport_checkpoint",
                input_semantics="independent_images",
                capabilities=ModelCapabilities(multiple_current_views=True),
                model_loader="keras_checkpoint",
                processor_loader="fixed_224",
                adapter="xreport_beit",
                output_sections=["raw_report"],
                max_current_images=16,
                anatomy_coverage="custom_training_data",
                coverage_note="Coverage depends on the data used by this XREPORT training run.",
                hardware_demand="moderate",
                parameter_label="Custom checkpoint",
                local_path=str(checkpoint.path),
                integrity_status="verified" if checkpoint.artifact_complete else "invalid",
                available_actions=["delete_local"],
            )
            for checkpoint in sorted(checkpoints, key=lambda item: item.name_key, reverse=True)
        ]
