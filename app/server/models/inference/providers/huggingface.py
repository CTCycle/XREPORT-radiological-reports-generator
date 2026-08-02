from __future__ import annotations

import gc
from io import BytesIO
from pathlib import Path
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from transformers import StoppingCriteria, StoppingCriteriaList

from server.configurations import InferenceSettings
from server.domain.inference import (
    GenerationProfile,
    InferenceImage,
    ProviderGenerationResult,
)
from server.models.inference.providers.adapters import ADAPTERS, StandardImageTextAdapter


REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ProgressCallback = Callable[
    [int, int, dict[str, str], list[dict[str, Any]] | None, dict[str, dict[str, str]] | None],
    None,
]

###############################################################################
class _InferenceStoppingCriteria(StoppingCriteria):
    """Stop generation cooperatively when cancellation or the deadline fires."""

    def __init__(self, should_stop: Callable[[], bool], deadline: float) -> None:
        self.should_stop = should_stop
        self.deadline = deadline

    # -------------------------------------------------------------------------
    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        del input_ids, scores, kwargs
        return self.should_stop() or time.monotonic() >= self.deadline

###############################################################################
class HuggingFaceProvider:
    """One-model, offline-only, manifest-driven Transformers runtime."""

    # -------------------------------------------------------------------------
    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._loaded_key: tuple[str, str, str, str, str] | None = None
        self._model: Any = None
        self._processor: Any = None
        self._adapter: StandardImageTextAdapter | None = None

    # -------------------------------------------------------------------------
    def is_cached(
        self,
        repository_id: str,
        revision: str | None,
        *,
        required_files: list[str] | None = None,
        weight_file_sets: list[list[str]] | None = None,
    ) -> bool:
        if not self.is_pinned_revision(revision) or not self.settings.hf_cache_dir:
            return False
        assert isinstance(revision, str)
        snapshot = self._snapshot_path(repository_id, revision)
        required = required_files or ["config.json"]
        if any(not (snapshot / path).is_file() for path in required):
            return False
        if weight_file_sets:
            if not any(
                all((snapshot / path).is_file() for path in alternatives)
                for alternatives in weight_file_sets
            ):
                return False
        return True

    # -------------------------------------------------------------------------
    def generate(
        self,
        *,
        repository_id: str,
        manifest: Mapping[str, Any],
        profile: GenerationProfile,
        clinical_context: str,
        images: list[InferenceImage],
        should_stop: Callable[[], bool],
        report_progress: ProgressCallback,
    ) -> ProviderGenerationResult:
        normalized = self.validate_manifest(repository_id, manifest)
        max_images = int(normalized["max_current_images"])
        if len(images) > max_images:
            raise ValueError(
                f"{repository_id} accepts at most {max_images} current image(s)"
            )
        if not images:
            raise ValueError("At least one image is required")
        if should_stop():
            return ProviderGenerationResult(
                reports={},
                display_sections={},
                metadata=[],
                provenance=self._provenance(normalized, profile, clinical_context),
            )

        deadline = time.monotonic() + self.settings.model_timeout
        with self._lock:
            if should_stop():
                return ProviderGenerationResult(
                    reports={},
                    display_sections={},
                    metadata=[],
                    provenance=self._provenance(normalized, profile, clinical_context),
                )
            model, processor, adapter = self._load(normalized)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{repository_id} inference exceeded the configured timeout")
            reports: dict[str, str] = {}
            display_sections: dict[str, dict[str, str]] = {}
            metadata: list[dict[str, Any]] = []
            for image_index, stored_image in enumerate(images, start=1):
                if should_stop() or time.monotonic() >= deadline:
                    break
                image, original_dimensions = self._decode_image(stored_image.data)
                prompt = adapter.prompt(profile, clinical_context)
                inputs, input_length = adapter.build_inputs(processor, image, prompt)
                processed_dimensions = adapter.processed_dimensions(inputs)
                inputs = self._move_inputs(inputs, model)
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        **adapter.generation_kwargs(profile),
                        stopping_criteria=StoppingCriteriaList([
                            _InferenceStoppingCriteria(should_stop, deadline),
                        ]),
                    )
                if should_stop():
                    return ProviderGenerationResult(
                        reports={},
                        display_sections={},
                        metadata=[],
                        provenance=self._provenance(normalized, profile, clinical_context),
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"{repository_id} inference exceeded the configured timeout"
                    )
                report = adapter.decode(processor, output, input_length=input_length)
                normalized_report = report.strip()
                if (
                    not normalized_report
                    or "\x00" in report
                    or not any(character.isalnum() for character in normalized_report)
                ):
                    raise RuntimeError(
                        f"{repository_id} returned an empty or malformed report for {stored_image.filename}"
                    )
                reports[stored_image.filename] = report
                display_sections[stored_image.filename] = adapter.display_sections(
                    report,
                    [str(section) for section in normalized.get("output_sections", ["raw_report"])],
                )
                metadata.append({
                    "filename": stored_image.filename,
                    "original_dimensions": {
                        "width": original_dimensions[0],
                        "height": original_dimensions[1],
                    },
                    "processed_tensor_dimensions": processed_dimensions,
                    "processor_loader": normalized["processor_loader"],
                    "model_loader": normalized["model_loader"],
                    "adapter": normalized["adapter"],
                    "prompt_profile": normalized.get("prompt_profile"),
                    "provider": "huggingface",
                    "model_ref": f"huggingface:{repository_id}",
                    "model_revision": normalized["revision"],
                    "generation_profile": profile,
                    "clinical_context": clinical_context,
                    "research_only": bool(normalized.get("research_only", True)),
                })
                report_progress(
                    image_index,
                    len(images),
                    reports,
                    metadata,
                    display_sections,
                )
            if should_stop():
                provenance = self._provenance(normalized, profile, clinical_context)
                provenance["input_images"] = []
                return ProviderGenerationResult(
                    reports={},
                    display_sections={},
                    metadata=[],
                    provenance=provenance,
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{repository_id} inference exceeded the configured timeout")
            if not reports:
                raise TimeoutError(f"{repository_id} inference exceeded the configured timeout")
            provenance = self._provenance(normalized, profile, clinical_context)
            provenance["input_images"] = metadata
            return ProviderGenerationResult(
                reports=reports,
                display_sections=display_sections,
                metadata=metadata,
                provenance=provenance,
            )

    # -------------------------------------------------------------------------
    def _load(
        self,
        manifest: Mapping[str, Any],
    ) -> tuple[Any, Any, StandardImageTextAdapter]:
        repository_id = str(manifest["repository_id"])
        revision = str(manifest["revision"])
        adapter_name = str(manifest["adapter"])
        processor_loader = str(manifest["processor_loader"])
        model_loader = str(manifest["model_loader"])
        key = (repository_id, revision, adapter_name, processor_loader, model_loader)
        with self._lock:
            if self._loaded_key == key and self._model is not None and self._adapter:
                return self._model, self._processor, self._adapter
            self.unload()
            snapshot_path = snapshot_download(
                repo_id=repository_id,
                revision=revision,
                cache_dir=self.settings.hf_cache_dir,
                local_files_only=True,
            )
            adapter_type = ADAPTERS[adapter_name]
            adapter = adapter_type()
            load_options = {
                "local_files_only": True,
                "trust_remote_code": bool(manifest["trust_remote_code"]),
            }
            processor = adapter.load_processor(
                snapshot_path,
                processor_loader=processor_loader,
                load_options=load_options,
            )
            model = adapter.load_model(
                snapshot_path,
                model_loader=model_loader,
                load_options={
                    **load_options,
                    "torch_dtype": self._dtype(str(manifest["preferred_dtype"])),
                    "device_map": self._device_map(),
                },
            )
            model.eval()
            self._loaded_key = key
            self._model = model
            self._processor = processor
            self._adapter = adapter
            return model, processor, adapter

    # -------------------------------------------------------------------------
    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            self._adapter = None
            self._loaded_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    def _snapshot_path(self, repository_id: str, revision: str) -> Path:
        cache = Path(self.settings.hf_cache_dir or "")
        return cache / f"models--{repository_id.replace('/', '--')}" / "snapshots" / revision

    # -------------------------------------------------------------------------
    def _device_map(self) -> str:
        if self.settings.device == "auto":
            return "auto"
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Hugging Face inference but is unavailable")
        return self.settings.device

    # -------------------------------------------------------------------------
    @staticmethod
    def _dtype(preferred_dtype: str) -> torch.dtype:
        if preferred_dtype == "float32":
            return torch.float32
        if preferred_dtype == "float16":
            return torch.float16
        if preferred_dtype == "bfloat16":
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    # -------------------------------------------------------------------------
    @staticmethod
    def _move_inputs(inputs: Any, model: Any) -> Any:
        device = getattr(model, "device", None)
        model_dtype = getattr(model, "dtype", None)
        if not isinstance(device, torch.device):
            return inputs.to(device) if device is not None and hasattr(inputs, "to") else inputs
        if isinstance(inputs, Mapping):
            moved_inputs = dict(inputs)
            for key, value in moved_inputs.items():
                if not isinstance(value, torch.Tensor):
                    continue
                dtype = model_dtype if value.is_floating_point() and isinstance(model_dtype, torch.dtype) else value.dtype
                moved_inputs[key] = value.to(device=device, dtype=dtype)
            return moved_inputs
        return inputs.to(device)

    # -------------------------------------------------------------------------
    @staticmethod
    def is_pinned_revision(revision: str | None) -> bool:
        return bool(revision and REVISION_PATTERN.fullmatch(revision))

    # -------------------------------------------------------------------------
    @classmethod
    def validate_manifest(
        cls,
        repository_id: str,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(manifest)
        normalized["repository_id"] = repository_id
        revision = normalized.get("revision")
        if not cls.is_pinned_revision(revision):
            raise RuntimeError(
                f"{repository_id} requires a pinned 40-character revision"
            )
        if normalized.get("trust_remote_code") and not normalized.get(
            "remote_code_approved", False
        ):
            raise RuntimeError(
                f"Remote code is not approved for pinned repository {repository_id}"
            )
        adapter_name = str(normalized.get("adapter", "standard_image_text"))
        if adapter_name not in ADAPTERS:
            raise RuntimeError(f"Unsupported Hugging Face adapter: {adapter_name}")
        normalized["adapter"] = adapter_name
        normalized["model_loader"] = str(normalized.get("model_loader", "image_text_to_text"))
        normalized["processor_loader"] = str(normalized.get("processor_loader", "auto"))
        if normalized["model_loader"] not in {
            "image_text_to_text",
            "causal_lm",
            "blip_conditional_generation",
        }:
            raise RuntimeError(
                f"Unsupported Transformers model loader: {normalized['model_loader']}"
            )
        if normalized["processor_loader"] not in {"auto", "image", "blip"}:
            raise RuntimeError(
                f"Unsupported Transformers processor loader: {normalized['processor_loader']}"
            )
        normalized["max_current_images"] = int(normalized.get("max_current_images", 1))
        if normalized["max_current_images"] < 1:
            raise RuntimeError("max_current_images must be at least 1")
        normalized["preferred_dtype"] = str(normalized.get("preferred_dtype", "auto"))
        if normalized["preferred_dtype"] not in {
            "auto", "float32", "float16", "bfloat16"
        }:
            raise RuntimeError(
                f"Unsupported preferred dtype: {normalized['preferred_dtype']}"
            )
        return normalized

    # -------------------------------------------------------------------------
    @staticmethod
    def _provenance(
        manifest: Mapping[str, Any],
        profile: GenerationProfile,
        clinical_context: str,
    ) -> dict[str, Any]:
        return {
            "provider": "huggingface",
            "model_ref": f"huggingface:{manifest['repository_id']}",
            "model_revision": manifest["revision"],
            "adapter": manifest["adapter"],
            "model_loader": manifest["model_loader"],
            "processor_loader": manifest["processor_loader"],
            "prompt_profile": manifest.get("prompt_profile"),
            "processor_repository_id": manifest.get("processor_repository_id"),
            "processor_revision": manifest.get("processor_revision"),
            "generation_profile": profile,
            "clinical_context": clinical_context,
            "research_only": bool(manifest.get("research_only", True)),
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _decode_image(data: bytes) -> tuple[Image.Image, tuple[int, int]]:
        try:
            with Image.open(BytesIO(data)) as decoded:
                oriented = ImageOps.exif_transpose(decoded)
                oriented.load()
                image = oriented.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed to decode inference image") from exc
        return image, image.size
