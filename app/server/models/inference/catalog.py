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
from server.models.inference.providers.ollama import OllamaProvider
from server.models.inference.providers.huggingface import HuggingFaceProvider


CATALOG_PATH = Path(__file__).resolve().parents[4] / "settings" / "inference_models.json"


class InferenceModelCatalog:
    """Lists only curated local models; catalog reads never download weights."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings

    def list_models(self) -> InferenceModelsResponse:
        installed_ollama = OllamaProvider(self.settings).installed_models()
        models = self._configured_models(installed_ollama, HuggingFaceProvider(self.settings))
        models.extend(self._xreport_models())
        huggingface_status = self._huggingface_provider_status()
        if any(model.provider == "huggingface" and model.status == "ready" for model in models):
            huggingface_status = ProviderAvailability(status="ready")
        return InferenceModelsResponse(
            models=models,
            providers={
                "ollama": self._ollama_provider_status(installed_ollama),
                "huggingface": huggingface_status,
                "xreport": ProviderAvailability(status="ready"),
            },
        )

    def _configured_models(self, installed_ollama: set[str] | None, huggingface: HuggingFaceProvider) -> list[ModelAvailability]:
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        configured = payload.get("models", [])
        if not isinstance(configured, list):
            raise ValueError("Inference model catalog must contain a models list")
        return [self._configured_model(entry, installed_ollama, huggingface) for entry in configured if isinstance(entry, dict)]

    def _configured_model(self, entry: dict[str, Any], installed_ollama: set[str] | None, huggingface: HuggingFaceProvider) -> ModelAvailability:
        provider = str(entry["provider"])
        status = "disabled" if provider == "huggingface" and not self.settings.hf_local_only else "not_installed"
        if provider == "ollama":
            status = "runtime_unavailable" if installed_ollama is None else ("ready" if entry["model_ref"].removeprefix("ollama:") in installed_ollama else "not_installed")
        if provider == "huggingface" and self.settings.hf_local_only and huggingface.is_cached(entry["model_ref"].removeprefix("huggingface:")):
            status = "ready"
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

    def _ollama_provider_status(self, installed_models: set[str] | None) -> ProviderAvailability:
        if installed_models is None:
            return ProviderAvailability(status="runtime_unavailable", message="Ollama is not running at the configured local address.")
        return ProviderAvailability(status="ready")

    def _huggingface_provider_status(self) -> ProviderAvailability:
        if not self.settings.hf_local_only:
            return ProviderAvailability(
                status="disabled",
                message="XREPORT_HF_LOCAL_ONLY must remain enabled for local inference.",
            )
        return ProviderAvailability(status="not_installed", message="No cached Hugging Face model has been discovered yet.")

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
