from __future__ import annotations

import gc
from io import BytesIO
from pathlib import Path
import re
import threading
from collections.abc import Callable, Mapping
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

from server.configurations import InferenceSettings
from server.domain.inference import GenerationProfile, InferenceImage
from server.models.inference.providers.adapters import ADAPTERS, StandardImageTextAdapter


REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ProgressCallback = Callable[
    [int, int, dict[str, str], list[dict[str, Any]] | None],
    None,
]

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
    def is_cached(self, repository_id: str, revision: str | None) -> bool:
        if not self.is_pinned_revision(revision) or not self.settings.hf_cache_dir:
            return False
        assert isinstance(revision, str)
        snapshot = self._snapshot_path(repository_id, revision)
        return (snapshot / "config.json").is_file()

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
    ) -> dict[str, str]:
        normalized = self.validate_manifest(repository_id, manifest)
        max_images = int(normalized["max_current_images"])
        if len(images) > max_images:
            raise ValueError(
                f"{repository_id} accepts at most {max_images} current image(s)"
            )
        if not images:
            raise ValueError("At least one image is required")
        if should_stop():
            return {}

        with self._lock:
            model, processor, adapter = self._load(normalized)
            reports: dict[str, str] = {}
            metadata: list[dict[str, Any]] = []
            for image_index, stored_image in enumerate(images, start=1):
                if should_stop():
                    break
                image, original_dimensions = self._decode_image(stored_image.data)
                prompt = adapter.prompt(profile, clinical_context)
                inputs, input_length = adapter.build_inputs(processor, image, prompt)
                processed_dimensions = adapter.processed_dimensions(inputs)
                if hasattr(inputs, "to"):
                    inputs = inputs.to(model.device)
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=self._max_new_tokens(profile),
                        do_sample=False,
                    )
                report = adapter.decode(processor, output, input_length=input_length)
                if not report:
                    raise RuntimeError(
                        f"{repository_id} returned an empty report for {stored_image.filename}"
                    )
                reports[stored_image.filename] = report
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
                })
                report_progress(image_index, len(images), reports, metadata)
            return reports

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
                    "dtype": self._dtype(str(manifest["preferred_dtype"])),
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
        if normalized["model_loader"] not in {"image_text_to_text", "causal_lm"}:
            raise RuntimeError(
                f"Unsupported Transformers model loader: {normalized['model_loader']}"
            )
        if normalized["processor_loader"] not in {"auto", "image"}:
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
    def _decode_image(data: bytes) -> tuple[Image.Image, tuple[int, int]]:
        try:
            with Image.open(BytesIO(data)) as decoded:
                oriented = ImageOps.exif_transpose(decoded)
                oriented.load()
                image = oriented.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed to decode inference image") from exc
        return image, image.size

    # -------------------------------------------------------------------------
    @staticmethod
    def _max_new_tokens(profile: GenerationProfile) -> int:
        return {"deterministic": 768, "concise": 384, "detailed": 1536}[profile]
