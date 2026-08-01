from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)

###############################################################################
class StandardImageTextAdapter:
    """Adapter for Transformers models that expose the standard multimodal API."""

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
        }
        loader = loaders.get(model_loader)
        if loader is None:
            raise ValueError(f"Unsupported Transformers model loader: {model_loader}")
        return loader.from_pretrained(snapshot_path, **load_options)

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
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
        input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
        input_length = int(input_ids.shape[-1]) if input_ids is not None else 0
        return inputs, input_length

    def decode(self, processor: Any, output: Any, *, input_length: int) -> str:
        generated = output[0][input_length:] if input_length else output[0]
        if hasattr(processor, "decode"):
            return str(processor.decode(generated, skip_special_tokens=True)).strip()
        if hasattr(processor, "batch_decode"):
            return str(processor.batch_decode(
                [generated], skip_special_tokens=True
            )[0]).strip()
        raise RuntimeError("Transformers processor cannot decode generated output")

    @staticmethod
    def prompt(profile: str, clinical_context: str) -> str:
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
    """MedGemma's standard processor with its model-specific report prompt."""

    @staticmethod
    def prompt(profile: str, clinical_context: str) -> str:
        base = StandardImageTextAdapter.prompt(profile, clinical_context)
        return f"{base}\nReturn explicit Findings and Impression sections."

###############################################################################
ADAPTERS: dict[str, type[StandardImageTextAdapter]] = {
    "standard_image_text": StandardImageTextAdapter,
    "medgemma": MedGemmaAdapter,
}
