from __future__ import annotations

import gc
from io import BytesIO
from pathlib import Path
import re
import threading
from collections.abc import Callable
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from server.configurations import InferenceSettings
from server.domain.inference import GenerationProfile, InferenceImage


REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


###############################################################################
class HuggingFaceProvider:
    """One-model, offline-only Hugging Face runtime."""

    # -------------------------------------------------------------------------
    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._loaded_key: tuple[str, str] | None = None
        self._model: Any = None
        self._processor: Any = None

    # -------------------------------------------------------------------------
    def is_cached(self, repository_id: str, revision: str | None = None) -> bool:
        pinned_revision = revision or self.settings.hf_medgemma_revision
        if (
            not isinstance(pinned_revision, str)
            or not self.is_pinned_revision(pinned_revision)
            or not self.settings.hf_cache_dir
        ):
            return False
        snapshot = self._snapshot_path(repository_id, pinned_revision)
        return (snapshot / "config.json").is_file()

    # -------------------------------------------------------------------------
    def generate(
        self,
        *,
        repository_id: str,
        revision: str,
        profile: GenerationProfile,
        clinical_context: str,
        images: list[InferenceImage],
        should_stop: Callable[[], bool],
        report_progress: Callable[[int, int, dict[str, str]], None],
    ) -> dict[str, str]:
        if not self.settings.hf_local_only:
            raise RuntimeError("Hugging Face generation requires local-only mode")
        if not self.is_pinned_revision(revision):
            raise RuntimeError("MedGemma requires a pinned 40-character revision")
        if len(images) != 1:
            raise ValueError("MedGemma accepts exactly one image")
        if should_stop():
            return {}

        model, processor = self._load(repository_id, revision)
        image = Image.open(BytesIO(images[0].data)).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self._prompt(profile, clinical_context)},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device, dtype=self._dtype())
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens(profile),
                do_sample=False,
            )
        report = processor.decode(
            output[0][input_length:],
            skip_special_tokens=True,
        ).strip()
        if not report:
            raise RuntimeError("MedGemma returned an empty report")
        if should_stop():
            return {}
        reports = {images[0].filename: report}
        report_progress(1, 1, reports)
        return reports

    # -------------------------------------------------------------------------
    def _load(self, repository_id: str, revision: str) -> tuple[Any, Any]:
        key = (repository_id, revision)
        with self._lock:
            if self._loaded_key == key and self._model is not None:
                return self._model, self._processor
            self.unload()
            snapshot_path = snapshot_download(
                repo_id=repository_id,
                revision=revision,
                cache_dir=self.settings.hf_cache_dir,
                local_files_only=True,
            )
            load_options = {
                "local_files_only": True,
                "trust_remote_code": False,
            }
            self._processor = AutoProcessor.from_pretrained(snapshot_path, **load_options)
            self._model = AutoModelForImageTextToText.from_pretrained(
                snapshot_path,
                dtype=self._dtype(),
                device_map=self._device_map(),
                **load_options,
            )
            self._model.eval()
            self._loaded_key = key
            return self._model, self._processor

    # -------------------------------------------------------------------------
    def unload(self) -> None:
        self._model = None
        self._processor = None
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
            raise RuntimeError("CUDA was requested for MedGemma but is unavailable")
        return self.settings.device

    # -------------------------------------------------------------------------
    def _dtype(self) -> torch.dtype:
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    # -------------------------------------------------------------------------
    @staticmethod
    def is_pinned_revision(revision: str | None) -> bool:
        return bool(revision and REVISION_PATTERN.fullmatch(revision))

    # -------------------------------------------------------------------------
    @staticmethod
    def _max_new_tokens(profile: GenerationProfile) -> int:
        return {"deterministic": 768, "concise": 384, "detailed": 1536}[profile]

    # -------------------------------------------------------------------------
    @staticmethod
    def _prompt(profile: GenerationProfile, clinical_context: str) -> str:
        detail = {
            "deterministic": "Use a consistent and conservative structure.",
            "concise": "Keep Findings and Impression concise.",
            "detailed": "Provide detailed Findings and Impression sections.",
        }[profile]
        context = clinical_context.strip() or "No clinical context supplied."
        return (
            "Draft a radiology report for research use only. The output is preliminary, "
            f"not clinically approved, and requires qualified review. {detail}\n"
            f"Clinical context: {context}"
        )
