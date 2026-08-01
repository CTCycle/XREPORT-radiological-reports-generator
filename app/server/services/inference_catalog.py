from __future__ import annotations

import json
from typing import Any

from server.common.path import SETTINGS_DIR
from server.configurations import InferenceSettings
from server.domain.inference import (
    InferenceModelsResponse,
    ModelAvailability,
    ModelCapabilities,
    ProviderAvailability,
)
from server.models.inference.providers.huggingface import HuggingFaceProvider
from server.repositories.serialization.model import ModelSerializer


CATALOG_PATH = SETTINGS_DIR / "inference_models.json"

###############################################################################
class InferenceModelCatalog:
    """Lists curated embedded models and discovered XREPORT checkpoints."""

    # -------------------------------------------------------------------------
    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings

    # -------------------------------------------------------------------------
    def list_models(self) -> InferenceModelsResponse:
        huggingface = HuggingFaceProvider(self.settings)
        models = self._configured_models(huggingface)
        xreport_models = self._xreport_models()
        models.extend(xreport_models)
        huggingface_status = self._huggingface_provider_status(models)
        if any(model.provider == "huggingface" and model.status == "ready" for model in models):
            huggingface_status = ProviderAvailability(status="ready")
        return InferenceModelsResponse(
            models=models,
            providers={
                "huggingface": huggingface_status,
                "xreport": self._xreport_provider_status(xreport_models),
            },
        )

    # -------------------------------------------------------------------------
    def _configured_models(
        self,
        huggingface: HuggingFaceProvider,
    ) -> list[ModelAvailability]:
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        configured = payload.get("models", [])
        if not isinstance(configured, list):
            raise ValueError("Inference model catalog must contain a models list")
        return [
            self._configured_model(entry, huggingface)
            for entry in configured
            if isinstance(entry, dict)
        ]

    # -------------------------------------------------------------------------
    def _configured_model(
        self,
        entry: dict[str, Any],
        huggingface: HuggingFaceProvider,
    ) -> ModelAvailability:
        provider = str(entry["provider"])
        if provider != "huggingface":
            raise ValueError(f"Unsupported configured inference provider: {provider}")

        revision = entry.get("revision")
        status = "disabled" if not self.settings.hf_local_only else "not_installed"
        status_message: str | None = None
        if not self.settings.hf_local_only:
            status_message = "HF_LOCAL_ONLY must remain enabled for local inference."
        else:
            try:
                huggingface.validate_manifest(
                    str(entry["model_ref"]).removeprefix("huggingface:"),
                    entry,
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                status = "incompatible"
                status_message = str(exc)
            if status_message is None and not huggingface.is_cached(
                str(entry["model_ref"]).removeprefix("huggingface:"), revision
            ):
                status_message = "The pinned snapshot is not available in the configured local cache."
            elif status_message is None and not bool(entry.get("validated", False)):
                status = "incompatible"
                status_message = (
                    "Candidate is not operational until a chest X-ray produces non-empty, "
                    "clinically structured report text."
                )
            elif status_message is None:
                status = "ready"

        capabilities = ModelCapabilities.model_validate(entry.get("capabilities", {}))
        return ModelAvailability.model_validate(
            {
                "model_ref": str(entry["model_ref"]),
                "provider": provider,
                "display_name": str(entry["display_name"]),
                "description": str(entry["description"]),
                "status": status,
                "status_message": status_message,
                "category": str(entry["category"]),
                "recommended": bool(entry.get("recommended", False)),
                "research_only": bool(entry.get("research_only", True)),
                "gated": bool(entry.get("gated", False)),
                "parameter_size": entry.get("parameter_size"),
                "local_size_bytes": entry.get("local_size_bytes"),
                "input_semantics": entry.get("input_semantics", "single_study"),
                "capabilities": capabilities,
                "model_revision": revision,
                "model_loader": entry.get("model_loader"),
                "processor_loader": entry.get("processor_loader"),
                "adapter": entry.get("adapter"),
                "trust_remote_code": bool(entry.get("trust_remote_code", False)),
                "remote_code_approved": bool(entry.get("remote_code_approved", False)),
                "output_sections": entry.get("output_sections", []),
                "max_current_images": int(entry.get("max_current_images", 1)),
                "supports_prior_images": bool(entry.get("supports_prior_images", False)),
                "supports_clinical_context": bool(entry.get("supports_clinical_context", False)),
                "preferred_dtype": str(entry.get("preferred_dtype", "auto")),
                "quantization": entry.get("quantization", ["none"]),
                "prompt_profile": entry.get("prompt_profile"),
                "license": entry.get("license"),
            }
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
        if all(model.status == "incompatible" for model in hf_models):
            return ProviderAvailability(
                status="incompatible",
                message=hf_models[0].status_message,
            )
        return ProviderAvailability(
            status="not_installed",
            message="No operational cached Hugging Face model has been validated yet.",
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _xreport_provider_status(
        models: list[ModelAvailability],
    ) -> ProviderAvailability:
        if models:
            return ProviderAvailability(status="ready")
        return ProviderAvailability(
            status="not_installed",
            message="No complete XREPORT checkpoints have been discovered yet.",
        )

    # -------------------------------------------------------------------------
    def _xreport_models(self) -> list[ModelAvailability]:
        checkpoint_names = ModelSerializer().scan_checkpoints_folder()
        return [
            ModelAvailability(
                model_ref=f"xreport:{checkpoint_name}",
                provider="xreport",
                display_name=checkpoint_name,
                description=(
                    "Local XREPORT checkpoint using the fixed BEiT 224x224x3 "
                    "image encoder"
                ),
                status="ready",
                category="xreport_checkpoint",
                input_semantics="independent_images",
                capabilities=ModelCapabilities(
                    multiple_current_views=True,
                    findings=True,
                    impression=True,
                ),
                model_loader="keras_checkpoint",
                processor_loader="fixed_224",
                adapter="xreport_beit",
                output_sections=["findings", "impression"],
                max_current_images=16,
            )
            for checkpoint_name in sorted(checkpoint_names, reverse=True)
        ]
