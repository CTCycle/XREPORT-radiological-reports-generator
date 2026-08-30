from __future__ import annotations

import gc
from io import BytesIO
from pathlib import Path
import re
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.utils.hub import HF_MODULES_CACHE

from server.configurations import InferenceSettings
from server.common.path import is_within_allowed_roots
from server.domain.inference import (
    GenerationProfile,
    InferenceImage,
    ProviderGenerationResult,
)
from server.common.inference_manifest import (
    is_pinned_revision,
    validate_manifest,
)
from server.models.inference.providers.adapters import (
    ADAPTERS,
    StandardImageTextAdapter,
    StudyGeneration,
    StudyImage,
)


ProgressCallback = Callable[
    [
        int,
        int,
        dict[str, str],
        list[dict[str, Any]] | None,
        dict[str, dict[str, str]] | None,
    ],
    None,
]


###############################################################################
class _InferenceStoppingCriteria(StoppingCriteria):
    """Stop generation cooperatively when cancellation or the deadline fires."""

    # -------------------------------------------------------------------------
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
    def generate(  # noqa: C901
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
        self._validate_images(repository_id, normalized, images)
        if should_stop():
            return self._empty_result(normalized, profile, clinical_context)

        deadline = time.monotonic() + self.settings.model_timeout
        with self._lock:
            if should_stop():
                return self._empty_result(
                    normalized,
                    profile,
                    clinical_context,
                )
            model, processor, adapter = self._load(normalized)
            self._check_deadline(repository_id, deadline)
            reports: dict[str, str] = {}
            display_sections: dict[str, dict[str, str]] = {}
            metadata: list[dict[str, Any]] = []
            if adapter.supports_study:
                generated = self._generate_study(
                    repository_id=repository_id,
                    normalized=normalized,
                    profile=profile,
                    clinical_context=clinical_context,
                    stored_images=images,
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    should_stop=should_stop,
                    deadline=deadline,
                )
                if generated is None:
                    return self._empty_result(normalized, profile, clinical_context)
                study_filename = images[0].filename
                reports[study_filename] = generated.report
                display_sections[study_filename] = generated.display_sections
                metadata.extend(generated.metadata)
                if should_stop():
                    return self._empty_result(
                        normalized,
                        profile,
                        clinical_context,
                        include_input_images=True,
                    )
                report_progress(
                    len(images),
                    len(images),
                    reports,
                    metadata,
                    display_sections,
                )
            for image_index, stored_image in enumerate(images, start=1):
                if adapter.supports_study:
                    break
                if should_stop() or time.monotonic() >= deadline:
                    break
                generated = self._generate_image(
                    repository_id=repository_id,
                    normalized=normalized,
                    profile=profile,
                    clinical_context=clinical_context,
                    stored_image=stored_image,
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    should_stop=should_stop,
                    deadline=deadline,
                )
                if generated is None:
                    return self._empty_result(normalized, profile, clinical_context)
                report, image_metadata, sections = generated
                if should_stop():
                    return self._empty_result(
                        normalized,
                        profile,
                        clinical_context,
                        include_input_images=True,
                    )
                reports[stored_image.filename] = report
                display_sections[stored_image.filename] = sections
                metadata.append(image_metadata)
                report_progress(
                    image_index,
                    len(images),
                    reports,
                    metadata,
                    display_sections,
                )
            if should_stop():
                return self._empty_result(
                    normalized,
                    profile,
                    clinical_context,
                    include_input_images=True,
                )
            self._check_deadline(repository_id, deadline)
            if not reports:
                raise TimeoutError(
                    f"{repository_id} inference exceeded the configured timeout"
                )
            provenance = self._provenance(normalized, profile, clinical_context)
            provenance["input_images"] = metadata
            provenance["report_scope"] = "study" if adapter.supports_study else "image"
            return ProviderGenerationResult(
                reports=reports,
                display_sections=display_sections,
                metadata=metadata,
                provenance=provenance,
            )

    # -------------------------------------------------------------------------
    def _generate_study(
        self,
        *,
        repository_id: str,
        normalized: Mapping[str, Any],
        profile: GenerationProfile,
        clinical_context: str,
        stored_images: list[InferenceImage],
        model: Any,
        processor: Any,
        adapter: StandardImageTextAdapter,
        should_stop: Callable[[], bool],
        deadline: float,
    ) -> StudyGeneration | None:
        if should_stop():
            return None
        study_images: list[StudyImage] = []
        for stored_image in stored_images:
            image, original_dimensions = self._decode_image(stored_image.data)
            study_images.append(
                StudyImage(
                    stored=stored_image,
                    image=image,
                    original_dimensions=original_dimensions,
                )
            )
        generated = adapter.generate_study(
            model=model,
            processor=processor,
            images=study_images,
            profile=profile,
            clinical_context=clinical_context,
            move_inputs=self._move_inputs,
            stopping_criteria=StoppingCriteriaList(
                [
                    _InferenceStoppingCriteria(should_stop, deadline),
                ]
            ),
            output_sections=[
                str(section)
                for section in normalized.get("output_sections", ["raw_report"])
            ],
        )
        if should_stop():
            return None
        self._check_deadline(repository_id, deadline)
        self._validate_report(repository_id, stored_images[0], generated.report)
        self._validate_display_sections(
            repository_id,
            stored_images[0],
            generated.display_sections,
            [
                str(section)
                for section in normalized.get("output_sections", ["raw_report"])
            ],
        )
        for item in generated.metadata:
            item.update(
                {
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
                }
            )
        return generated

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_images(
        repository_id: str,
        manifest: Mapping[str, Any],
        images: list[InferenceImage],
    ) -> None:
        max_images = int(manifest["max_current_images"])
        if len(images) > max_images:
            raise ValueError(
                f"{repository_id} accepts at most {max_images} current image(s)"
            )
        if not images:
            raise ValueError("At least one image is required")

    # -------------------------------------------------------------------------
    @staticmethod
    def _empty_result(
        manifest: Mapping[str, Any],
        profile: GenerationProfile,
        clinical_context: str,
        *,
        include_input_images: bool = False,
    ) -> ProviderGenerationResult:
        provenance = HuggingFaceProvider._provenance(
            manifest, profile, clinical_context
        )
        if include_input_images:
            provenance["input_images"] = []
        return ProviderGenerationResult(
            reports={},
            display_sections={},
            metadata=[],
            provenance=provenance,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_deadline(repository_id: str, deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{repository_id} inference exceeded the configured timeout"
            )

    # -------------------------------------------------------------------------
    def _generate_image(
        self,
        *,
        repository_id: str,
        normalized: Mapping[str, Any],
        profile: GenerationProfile,
        clinical_context: str,
        stored_image: InferenceImage,
        model: Any,
        processor: Any,
        adapter: StandardImageTextAdapter,
        should_stop: Callable[[], bool],
        deadline: float,
    ) -> tuple[str, dict[str, Any], dict[str, str]] | None:
        image, original_dimensions = self._decode_image(stored_image.data)
        prompt = adapter.prompt(profile, clinical_context)
        inputs, input_length = adapter.build_inputs(processor, image, prompt)
        processed_dimensions = adapter.processed_dimensions(inputs)
        inputs = self._move_inputs(inputs, model)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                **adapter.generation_kwargs(profile),
                stopping_criteria=StoppingCriteriaList(
                    [
                        _InferenceStoppingCriteria(should_stop, deadline),
                    ]
                ),
            )
        if should_stop():
            return None
        self._check_deadline(repository_id, deadline)
        report = adapter.decode(processor, output, input_length=input_length)
        self._validate_report(repository_id, stored_image, report)
        metadata = {
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
        }
        sections = adapter.display_sections(
            report,
            [
                str(section)
                for section in normalized.get("output_sections", ["raw_report"])
            ],
        )
        self._validate_display_sections(
            repository_id,
            stored_image,
            sections,
            [
                str(section)
                for section in normalized.get("output_sections", ["raw_report"])
            ],
        )
        return report, metadata, sections

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_report(
        repository_id: str,
        stored_image: InferenceImage,
        report: str,
    ) -> None:
        normalized_report = report.strip()
        if (
            not normalized_report
            or "\x00" in report
            or not any(character.isalnum() for character in normalized_report)
        ):
            raise RuntimeError(
                f"{repository_id} returned an empty or malformed report for {stored_image.filename}"
            )
        words = re.findall(r"[A-Za-z0-9]+", normalized_report.lower())
        if len(words) >= 24 and len(set(words)) / len(words) < 0.12:
            raise RuntimeError(
                f"{repository_id} returned a pathologically repetitive report for {stored_image.filename}"
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_display_sections(
        repository_id: str,
        stored_image: InferenceImage,
        sections: Mapping[str, str],
        output_sections: list[str],
    ) -> None:
        if any(
            not str(sections.get(section, "")).strip() for section in output_sections
        ):
            raise RuntimeError(
                f"{repository_id} returned incomplete report sections for {stored_image.filename}"
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
            configured_snapshot = manifest.get("local_snapshot_path")
            if configured_snapshot:
                snapshot_path = Path(str(configured_snapshot)).resolve()
                if not is_within_allowed_roots(snapshot_path):
                    raise RuntimeError(
                        "Model snapshot must be inside application storage"
                    )
                if not snapshot_path.is_dir():
                    raise RuntimeError(
                        f"Verified local model snapshot is missing: {snapshot_path}"
                    )
            else:
                raise RuntimeError("Verified local model snapshot path is missing")
            adapter_type = ADAPTERS[adapter_name]
            adapter = adapter_type()
            load_options = {
                "local_files_only": True,
                "trust_remote_code": bool(manifest["trust_remote_code"]),
            }
            if load_options["trust_remote_code"]:
                self._prepare_verified_remote_code_cache(
                    snapshot_path,
                    revision,
                    manifest.get("required_files", []),
                )
            processor = adapter.load_processor(
                str(snapshot_path),
                processor_loader=processor_loader,
                load_options=load_options,
            )
            model = adapter.load_model(
                str(snapshot_path),
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
    @staticmethod
    def _prepare_verified_remote_code_cache(
        snapshot_path: Path,
        revision: str,
        required_files: Any,
    ) -> None:
        """Populate Transformers' dynamic-module cache from verified files only.

        Transformers copies relative remote-code imports into a separate
        dynamic-module cache. Some older model cards omit transitive imports
        during that copy, which later causes a local-only load to fail. The
        manifest is the allowlist: only its verified remote-code files and the
        small JSON assets they load relative to ``__file__`` are copied, and no
        Hub lookup or user-level cache is consulted.
        """
        cache_root = Path(HF_MODULES_CACHE).resolve()
        if not is_within_allowed_roots(cache_root):
            raise RuntimeError(
                "Transformers remote-code cache must remain inside application storage"
            )
        module_path = cache_root / "transformers_modules" / f"_{revision}"
        module_path.mkdir(parents=True, exist_ok=True)
        init_file = module_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
        for filename in required_files:
            relative = Path(str(filename))
            if (
                relative.suffix not in {".py", ".json"}
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                continue
            source = (snapshot_path / relative).resolve()
            try:
                source.relative_to(snapshot_path.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    "Remote-code manifest points outside the verified snapshot"
                ) from exc
            if not source.is_file():
                raise RuntimeError(
                    f"Verified remote-code file is missing: {relative.as_posix()}"
                )
            destination = module_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or not destination.samefile(source):
                shutil.copyfile(source, destination)

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
    def _device_map(self) -> str:
        if self.settings.device == "auto":
            return "auto"
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for Hugging Face inference but is unavailable"
            )
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
            return (
                inputs.to(device)
                if device is not None and hasattr(inputs, "to")
                else inputs
            )
        if isinstance(inputs, Mapping):
            moved_inputs = dict(inputs)
            for key, value in moved_inputs.items():
                if not isinstance(value, torch.Tensor):
                    continue
                dtype = (
                    model_dtype
                    if value.is_floating_point()
                    and isinstance(model_dtype, torch.dtype)
                    else value.dtype
                )
                moved_inputs[key] = value.to(device=device, dtype=dtype)
            return moved_inputs
        return inputs.to(device)

    # -------------------------------------------------------------------------
    @staticmethod
    def is_pinned_revision(revision: str | None) -> bool:
        return is_pinned_revision(revision)

    # -------------------------------------------------------------------------
    @classmethod
    def validate_manifest(
        cls,
        repository_id: str,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        return validate_manifest(repository_id, manifest)

    # -------------------------------------------------------------------------
    @staticmethod
    def _provenance(
        manifest: Mapping[str, Any],
        profile: GenerationProfile,
        clinical_context: str,
    ) -> dict[str, Any]:
        provenance: dict[str, Any] = {
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
        validation_status = manifest.get("validation_status")
        validation_message = manifest.get("validation_message")
        if validation_status is not None:
            provenance["validation_status"] = validation_status
        if validation_message:
            provenance["validation_message"] = validation_message
        if validation_status == "degraded":
            provenance["quality_warnings"] = ["sensitivity_canary_failed"]
        return provenance

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
