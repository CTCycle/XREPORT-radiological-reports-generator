from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from server.domain.inference import GenerationProfile


###############################################################################
class StandardImageTextAdapter:
    """Adapter for standard Transformers multimodal chat-style models."""

    # -------------------------------------------------------------------------
    def load_processor(
        self,
        snapshot_path: str,
        *,
        processor_loader: str,
        load_options: dict[str, Any],
    ) -> Any:
        if processor_loader == "auto":
            return AutoProcessor.from_pretrained(snapshot_path, **load_options)
        if processor_loader == "image":
            return AutoImageProcessor.from_pretrained(snapshot_path, **load_options)
        raise ValueError(f"Unsupported Transformers processor loader: {processor_loader}")

    # -------------------------------------------------------------------------
    def load_model(
        self,
        snapshot_path: str,
        *,
        model_loader: str,
        load_options: dict[str, Any],
    ) -> Any:
        loaders = {
            "image_text_to_text": AutoModelForImageTextToText,
            "causal_lm": AutoModelForCausalLM,
            "blip_conditional_generation": BlipForConditionalGeneration,
        }
        loader = loaders.get(model_loader)
        if loader is None:
            raise ValueError(f"Unsupported Transformers model loader: {model_loader}")
        return loader.from_pretrained(snapshot_path, **load_options)

    # -------------------------------------------------------------------------
    def build_inputs(
        self,
        processor: Any,
        image: Image.Image,
        prompt: str,
    ) -> tuple[Any, int]:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        if hasattr(processor, "apply_chat_template"):
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(images=image, text=prompt, return_tensors="pt")
        input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
        input_length = int(input_ids.shape[-1]) if input_ids is not None else 0
        return inputs, input_length

    # -------------------------------------------------------------------------
    def decode(self, processor: Any, output: Any, *, input_length: int) -> str:
        generated = output[0][input_length:] if input_length else output[0]
        if hasattr(processor, "decode"):
            return str(processor.decode(generated, skip_special_tokens=True))
        if hasattr(processor, "batch_decode"):
            return str(processor.batch_decode(
                [generated], skip_special_tokens=True
            )[0])
        raise RuntimeError("Transformers processor cannot decode generated output")

    # -------------------------------------------------------------------------
    @staticmethod
    def prompt(profile: GenerationProfile, clinical_context: str) -> str:
        detail = {
            "deterministic": "Use a consistent and conservative structure.",
            "concise": "Keep the report concise.",
            "detailed": "Provide detailed observations while remaining clinically cautious.",
        }[profile]
        context = clinical_context.strip() or "No clinical context supplied."
        return (
            "Draft a radiology report for research use only. The output is preliminary, "
            "not clinically approved, and requires qualified review. "
            f"{detail}\nClinical context: {context}"
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def generation_kwargs(profile: GenerationProfile) -> dict[str, Any]:
        return {
            "max_new_tokens": {"deterministic": 768, "concise": 384, "detailed": 1536}[profile],
            "do_sample": False,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def display_sections(report: str, output_sections: list[str]) -> dict[str, str]:
        if "raw_report" in output_sections:
            return {"raw_report": report}

        sections: dict[str, str] = {}
        normalized = report.strip()
        if len(output_sections) == 1 and output_sections[0] in {"findings", "impression"}:
            sections[output_sections[0]] = normalized
            return sections

        matches = {
            section: re.search(
                rf"(?:^|\n)\s*(?:#{{1,3}}\s*)?{section}\s*:?\s*([\s\S]*?)(?=\n\s*(?:#{{1,3}}\s*)?(?:findings|impression)\s*:?|$)",
                normalized,
                flags=re.IGNORECASE,
            )
            for section in output_sections
        }
        for section, match in matches.items():
            if match:
                sections[section] = match.group(1).strip()
        return sections

    # -------------------------------------------------------------------------
    @staticmethod
    def processed_dimensions(inputs: Any) -> list[int] | None:
        if not isinstance(inputs, Mapping):
            return None
        pixel_values = inputs.get("pixel_values")
        shape = getattr(pixel_values, "shape", None)
        if shape is None:
            return None
        return [int(dimension) for dimension in shape]


###############################################################################
class MedGemmaAdapter(StandardImageTextAdapter):
    """MedGemma's standard processor with a research-only report prompt."""


###############################################################################
class BlipCxrAdapter(StandardImageTextAdapter):
    """BLIP conditional generation contract used by generate-cxr."""

    # -------------------------------------------------------------------------
    def load_processor(
        self,
        snapshot_path: str,
        *,
        processor_loader: str,
        load_options: dict[str, Any],
    ) -> Any:
        if processor_loader != "blip":
            raise ValueError(f"BLIP requires the blip processor loader, got {processor_loader}")
        return BlipProcessor.from_pretrained(snapshot_path, **load_options)

    # -------------------------------------------------------------------------
    def build_inputs(
        self,
        processor: Any,
        image: Image.Image,
        prompt: str,
    ) -> tuple[Any, int]:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
        input_length = int(input_ids.shape[-1]) if input_ids is not None else 0
        return inputs, input_length

    # -------------------------------------------------------------------------
    @staticmethod
    def prompt(profile: GenerationProfile, clinical_context: str) -> str:
        del profile
        return f"indication:{clinical_context.strip()}"

    # -------------------------------------------------------------------------
    @staticmethod
    def generation_kwargs(profile: GenerationProfile) -> dict[str, Any]:
        return {
            "max_length": {"deterministic": 512, "concise": 256, "detailed": 512}[profile],
            "do_sample": False,
        }

    # -------------------------------------------------------------------------
    def decode(self, processor: Any, output: Any, *, input_length: int) -> str:
        del input_length
        if not hasattr(processor, "decode"):
            raise RuntimeError("BLIP processor cannot decode generated output")
        return str(processor.decode(output[0], skip_special_tokens=True))


ADAPTERS: dict[str, type[StandardImageTextAdapter]] = {
    "standard_image_text": StandardImageTextAdapter,
    "medgemma": MedGemmaAdapter,
    "generate_cxr_blip": BlipCxrAdapter,
}
