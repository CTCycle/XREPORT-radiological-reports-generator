from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from server.configurations import InferenceSettings
from server.domain.inference import (
    InferenceModelsResponse,
    ModelAvailability,
    ModelCapabilities,
    ProviderAvailability,
)
from server.repositories.serialization.model import ModelSerializer


CATALOG_PATH = Path(__file__).resolve().parents[4] / "settings" / "inference_models.json"


class InferenceModelCatalog:
    """Lists only curated local models; catalog reads never download weights."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings

    def list_models(self) -> InferenceModelsResponse:
        models = self._configured_models()
        models.extend(self._xreport_models())
        return InferenceModelsResponse(
            models=models,
            providers={
                "ollama": ProviderAvailability(
                    status="runtime_unavailable",
                    message="Ollama discovery is enabled in the next provider increment.",
                ),
                "huggingface": self._huggingface_provider_status(),
                "xreport": ProviderAvailability(status="ready"),
            },
        )

    def _configured_models(self) -> list[ModelAvailability]:
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        configured = payload.get("models", [])
        if not isinstance(configured, list):
            raise ValueError("Inference model catalog must contain a models list")
        return [self._configured_model(entry) for entry in configured if isinstance(entry, dict)]

    def _configured_model(self, entry: dict[str, Any]) -> ModelAvailability:
        provider = str(entry["provider"])
        status = "disabled" if provider == "huggingface" and not self.settings.hf_local_only else "not_installed"
        return ModelAvailability(
            model_ref=str(entry["model_ref"]),
            provider=provider,  # type: ignore[arg-type]
            display_name=str(entry["display_name"]),
            description=str(entry["description"]),
            status=status,  # type: ignore[arg-type]
            category=str(entry["category"]),
            recommended=bool(entry.get("recommended", False)),
            research_only=bool(entry.get("research_only", True)),
            gated=bool(entry.get("gated", False)),
            parameter_size=entry.get("parameter_size"),
            local_size_bytes=entry.get("local_size_bytes"),
            input_semantics=entry.get("input_semantics", "single_image"),  # type: ignore[arg-type]
            capabilities=ModelCapabilities.model_validate(entry.get("capabilities", {})),
            model_revision=entry.get("model_revision"),
        )

    def _huggingface_provider_status(self) -> ProviderAvailability:
        if not self.settings.hf_local_only:
            return ProviderAvailability(
                status="disabled",
                message="XREPORT_HF_LOCAL_ONLY must remain enabled for local inference.",
            )
        return ProviderAvailability(
            status="not_installed",
            message="No cached Hugging Face model has been discovered yet.",
        )

    def _xreport_models(self) -> list[ModelAvailability]:
        checkpoint_names = ModelSerializer().scan_checkpoints_folder()
        return [
            ModelAvailability(
                model_ref=f"xreport:{checkpoint_name}",
                provider="xreport",
                display_name=checkpoint_name,
                description="Local XREPORT trained checkpoint",
                status="ready",
                category="xreport_checkpoint",
                input_semantics="independent_images",
                capabilities=ModelCapabilities(),
            )
            for checkpoint_name in sorted(checkpoint_names, reverse=True)
        ]
