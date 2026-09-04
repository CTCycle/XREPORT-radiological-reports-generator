from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from collections.abc import Mapping
from typing import Any, Callable, cast

from qwen_vl_utils import process_vision_info
from safetensors.torch import load_file
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils.hub import HF_MODULES_CACHE

from server.common.path import is_within_allowed_roots
from server.domain.inference import GenerationProfile, InferenceImage


###############################################################################
@dataclass(frozen=True)
class StudyImage:
    stored: InferenceImage
    image: Image.Image
    original_dimensions: tuple[int, int]


###############################################################################
@dataclass(frozen=True)
class StudyGeneration:
    report: str
    display_sections: dict[str, str]
    metadata: list[dict[str, Any]]


MoveInputs = Callable[[Any, Any], Any]
StoppingCriteriaValue = Any


###############################################################################
class _LegacyDecoderPrepareInputs:
    """Callable compatibility wrapper for CXRMate Multi's old decoder API."""

    # -------------------------------------------------------------------------
    def __init__(self, original: Callable[..., Any]) -> None:
        self.original = original
        self.__wrapped__ = original

    # -------------------------------------------------------------------------
    def __call__(
        self,
        input_ids: Any,
        *args: Any,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Any:
        if past_key_values is not None and kwargs.get("cache_position") is None:
            try:
                cached_length = int(past_key_values.get_seq_length())
            except (AttributeError, TypeError, ValueError):
                cached_length = max(int(getattr(input_ids, "shape", [1, 1])[-1]) - 1, 0)
            kwargs["cache_position"] = torch.tensor(
                [cached_length],
                device=getattr(input_ids, "device", None),
                dtype=torch.long,
            )
        return self.original(
            input_ids, *args, past_key_values=past_key_values, **kwargs
        )


###############################################################################
class _LegacyCacheView:
    """Shape-only legacy view used by the archived CXRMate-ED decoder."""

    # -------------------------------------------------------------------------
    def __init__(
        self, past_key_values: Any, input_ids: Any, inferred_length: int
    ) -> None:
        self.past_key_values = past_key_values
        self.input_ids = input_ids
        self.inferred_length = inferred_length

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.past_key_values)

    # -------------------------------------------------------------------------
    def __getitem__(self, index: int) -> Any:
        value = self.past_key_values[index]
        if index == 0 and isinstance(value, (tuple, list)) and value:
            fake = torch.empty(
                1,
                1,
                self.inferred_length,
                1,
                device=getattr(self.input_ids, "device", None),
            )
            return (fake, *value[1:])
        return value

    # -------------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self.past_key_values, name)


###############################################################################
class _CXRMateEDPrepareInputs:
    """Callable compatibility wrapper for CXRMate-ED's archived cache API."""

    # -------------------------------------------------------------------------
    def __init__(self, original: Callable[..., Any]) -> None:
        self.original = original
        self.__wrapped__ = original

    # -------------------------------------------------------------------------
    def __call__(
        self,
        input_ids: Any,
        *args: Any,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Any:
        if past_key_values is None:
            return self.original(
                input_ids, *args, past_key_values=past_key_values, **kwargs
            )
        try:
            first_layer = past_key_values[0]
            first_key = (
                first_layer[0] if isinstance(first_layer, (tuple, list)) else None
            )
        except (IndexError, KeyError, TypeError):
            first_key = None
        if first_key is not None:
            return self.original(
                input_ids, *args, past_key_values=past_key_values, **kwargs
            )

        inferred_length = max(int(getattr(input_ids, "shape", [1, 1])[-1]) - 1, 0)
        legacy_cache = _LegacyCacheView(past_key_values, input_ids, inferred_length)
        result = self.original(input_ids, *args, past_key_values=legacy_cache, **kwargs)
        if isinstance(result, dict):
            result["past_key_values"] = past_key_values
        return result


###############################################################################
def _ensure_legacy_decoder_cache_compatibility(model: Any) -> None:
    """Bridge old remote-code decoders to Transformers' cache-position API.

    CXRMate Multi's published ``prepare_inputs_for_generation`` forwards only
    ``past_key_values``. Transformers 4.57+ requires a cache position when a
    cache is present, so the decoder otherwise receives ``None`` and fails
    before the first generated token. This instance-local bridge leaves the
    verified snapshot untouched and is safe to apply once per loaded model.
    """
    decoder = getattr(model, "decoder", None)
    original = getattr(decoder, "prepare_inputs_for_generation", None)
    if (
        decoder is None
        or not callable(original)
        or getattr(decoder, "_xreport_cache_compat", False)
    ):
        return

    decoder.prepare_inputs_for_generation = _LegacyDecoderPrepareInputs(original)
    decoder._xreport_cache_compat = True


###############################################################################
def _ensure_cxrmate_ed_cache_compatibility(model: Any) -> None:  # noqa: C901
    """Keep the published CXRMate-ED decoder compatible with DynamicCache.

    The published remote code expects a legacy tuple whose first layer contains
    a key tensor. Transformers 4.57 can hand it a cache whose first layer is
    ``(None, None)`` while still carrying the usable cache object. We provide
    the decoder with a shape-only view for its prefix-length calculation and
    put the original cache back into the returned model inputs.
    """
    if getattr(model, "_xreport_ed_cache_compat", False):
        return
    original = getattr(model, "prepare_inputs_for_generation", None)
    if not callable(original):
        return

    model.prepare_inputs_for_generation = _CXRMateEDPrepareInputs(original)
    model._xreport_ed_cache_compat = True


###############################################################################
class StandardImageTextAdapter:
    """Adapter for standard Transformers multimodal chat-style models."""

    supports_study = False

    # -------------------------------------------------------------------------
    def generate_study(
        self,
        *,
        model: Any,
        processor: Any,
        images: list[StudyImage],
        profile: GenerationProfile,
        clinical_context: str,
        move_inputs: MoveInputs,
        stopping_criteria: StoppingCriteriaValue,
        output_sections: list[str],
    ) -> StudyGeneration:
        del model, processor, images, profile, clinical_context
        del move_inputs, stopping_criteria, output_sections
        raise NotImplementedError("This adapter generates one image at a time")

    # -------------------------------------------------------------------------
    def load_processor(
        self,
        snapshot_path: str,
        *,
        processor_loader: str,
        load_options: dict[str, Any],
    ) -> Any:
        if processor_loader != "auto":
            raise ValueError(
                f"Unsupported Transformers processor loader: {processor_loader}"
            )
        return AutoProcessor.from_pretrained(snapshot_path, **load_options)

    # -------------------------------------------------------------------------
    def load_model(
        self,
        snapshot_path: str,
        *,
        model_loader: str,
        load_options: dict[str, Any],
    ) -> Any:
        loaders = {
            "auto_model": AutoModel,
            "image_text_to_text": AutoModelForImageTextToText,
            "causal_lm": AutoModelForCausalLM,
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
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
            return str(processor.batch_decode([generated], skip_special_tokens=True)[0])
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
            "max_new_tokens": {"deterministic": 768, "concise": 384, "detailed": 1536}[
                profile
            ],
            "do_sample": False,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def display_sections(report: str, output_sections: list[str]) -> dict[str, str]:
        if "raw_report" in output_sections:
            return {"raw_report": report}

        sections: dict[str, str] = {}
        normalized = report.strip()
        if len(output_sections) == 1 and output_sections[0] in {
            "findings",
            "impression",
        }:
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
class ChatVisionStudyAdapter(StandardImageTextAdapter):
    """Study adapter for image-text-to-text models using chat templates."""

    supports_study = True

    # -------------------------------------------------------------------------
    def generate_study(
        self,
        *,
        model: Any,
        processor: Any,
        images: list[StudyImage],
        profile: GenerationProfile,
        clinical_context: str,
        move_inputs: MoveInputs,
        stopping_criteria: StoppingCriteriaValue,
        output_sections: list[str],
    ) -> StudyGeneration:
        prompt = self.prompt(profile, clinical_context)
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": item.image} for item in images),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if not hasattr(processor, "apply_chat_template"):
            raise RuntimeError(
                "The selected vision-language processor has no chat template"
            )
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
        input_length = int(input_ids.shape[-1]) if input_ids is not None else 0
        inputs = move_inputs(inputs, model)
        output = model.generate(
            **inputs,
            **self.generation_kwargs(profile),
            stopping_criteria=stopping_criteria,
        )
        report = self.decode(processor, output, input_length=input_length).strip()
        return StudyGeneration(
            report=report,
            display_sections=self.display_sections(report, output_sections),
            metadata=[self._metadata(item, processor, inputs) for item in images],
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _metadata(item: StudyImage, processor: Any, inputs: Any) -> dict[str, Any]:
        return {
            "filename": item.stored.filename,
            "original_dimensions": {
                "width": item.original_dimensions[0],
                "height": item.original_dimensions[1],
            },
            "processed_tensor_dimensions": StandardImageTextAdapter.processed_dimensions(
                inputs
            ),
            "processor_loader": "auto",
            "input_scope": "study",
        }


###############################################################################
class CheXOneAdapter(ChatVisionStudyAdapter):
    """CheXOne's Qwen2.5-VL processor with the published multi-image path."""

    # -------------------------------------------------------------------------
    def load_processor(
        self,
        snapshot_path: str,
        *,
        processor_loader: str,
        load_options: dict[str, Any],
    ) -> Any:
        return AutoProcessor.from_pretrained(
            snapshot_path,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 512,
            **load_options,
        )

    # -------------------------------------------------------------------------
    def generate_study(self, **kwargs: Any) -> StudyGeneration:
        model = kwargs["model"]
        processor = kwargs["processor"]
        images: list[StudyImage] = kwargs["images"]
        profile: GenerationProfile = kwargs["profile"]
        clinical_context = kwargs["clinical_context"]
        move_inputs: MoveInputs = kwargs["move_inputs"]
        stopping_criteria = kwargs["stopping_criteria"]
        output_sections: list[str] = kwargs["output_sections"]
        prompt = (
            "Write a radiology report for the supplied chest radiograph study. "
            "Return only the final Findings and Impression sections; do not include reasoning traces. "
            f"{self.prompt(profile, clinical_context)}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": item.image} for item in images),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        vision_info = process_vision_info(messages)
        image_inputs, video_inputs = vision_info[0], vision_info[1]
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        input_length = int(inputs.input_ids.shape[-1])
        inputs = move_inputs(inputs, model)
        output = model.generate(
            **inputs,
            max_new_tokens={"deterministic": 768, "concise": 384, "detailed": 1024}[
                profile
            ],
            do_sample=False,
            stopping_criteria=stopping_criteria,
        )
        generated_ids = [out_ids[input_length:] for out_ids in output]
        report = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return StudyGeneration(
            report=report,
            display_sections=self.display_sections(report, output_sections),
            metadata=[self._metadata(item, processor, inputs) for item in images],
        )


###############################################################################
class CXRMateMultiAdapter(StandardImageTextAdapter):
    """Published CXRMate multi-view encoder-decoder preprocessing contract."""

    supports_study = True

    # -------------------------------------------------------------------------
    def load_processor(
        self,
        snapshot_path: str,
        *,
        processor_loader: str,
        load_options: dict[str, Any],
    ) -> Any:
        return {
            "tokenizer": AutoTokenizer.from_pretrained(snapshot_path, **load_options),
            "image_processor": AutoFeatureExtractor.from_pretrained(
                snapshot_path, **load_options
            ),
        }

    # -------------------------------------------------------------------------
    def load_model(
        self,
        snapshot_path: str,
        *,
        model_loader: str,
        load_options: dict[str, Any],
    ) -> Any:
        return AutoModel.from_pretrained(snapshot_path, **load_options)

    # -------------------------------------------------------------------------
    def generate_study(
        self,
        *,
        model: Any,
        processor: Any,
        images: list[StudyImage],
        profile: GenerationProfile,
        clinical_context: str,
        move_inputs: MoveInputs,
        stopping_criteria: StoppingCriteriaValue,
        output_sections: list[str],
    ) -> StudyGeneration:
        del profile, clinical_context, stopping_criteria
        image_processor = processor["image_processor"]
        tokenizer = processor["tokenizer"]
        shortest_edge = int(image_processor.size["shortest_edge"])
        image_transform = transforms.Compose(
            [
                transforms.Resize(shortest_edge),
                transforms.CenterCrop((shortest_edge, shortest_edge)),
                transforms.ToTensor(),
                transforms.Normalize(
                    image_processor.image_mean, image_processor.image_std
                ),
            ]
        )
        tensors: list[torch.Tensor] = [
            cast(torch.Tensor, image_transform(item.image)) for item in images
        ]
        batch = torch.stack(tensors, dim=0).unsqueeze(0)
        moved = move_inputs({"pixel_values": batch}, model)
        _ensure_legacy_decoder_cache_compatibility(model)
        output = model.generate(
            pixel_values=moved["pixel_values"],
            special_token_ids=[tokenizer.sep_token_id],
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
            max_length=256,
            num_beams=4,
        )
        findings, impression = model.split_and_decode_sections(
            output.sequences,
            [tokenizer.sep_token_id, tokenizer.eos_token_id],
            tokenizer,
        )
        report = f"Findings:\n{findings[0].strip()}\n\nImpression:\n{impression[0].strip()}".strip()
        return StudyGeneration(
            report=report,
            display_sections={
                "findings": findings[0].strip(),
                "impression": impression[0].strip(),
            },
            metadata=[self._metadata(item, moved["pixel_values"]) for item in images],
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _metadata(item: StudyImage, tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "filename": item.stored.filename,
            "original_dimensions": {
                "width": item.original_dimensions[0],
                "height": item.original_dimensions[1],
            },
            "processed_tensor_dimensions": [int(value) for value in tensor.shape],
            "processor_loader": "auto_feature_extractor",
            "input_scope": "study",
        }


###############################################################################
class CXRMateEDAdapter(StandardImageTextAdapter):
    """Published CXRMate-ED image-only/context-aware study preprocessing."""

    supports_study = True
    generation_profiles: dict[GenerationProfile, dict[str, Any]] = {
        "deterministic": {
            "max_length": 256,
            "num_beams": 1,
            "do_sample": False,
        },
        "concise": {
            "max_length": 160,
            "num_beams": 1,
            "do_sample": False,
        },
        "detailed": {
            "max_length": 384,
            "num_beams": 4,
            "do_sample": False,
        },
    }

    # -------------------------------------------------------------------------
    def load_model(
        self,
        snapshot_path: str,
        *,
        model_loader: str,
        load_options: dict[str, Any],
    ) -> Any:
        del model_loader
        # CXRMate-ED predates the current Llava base-model prefix convention and
        # references a second custom repository for its UniFormer image encoder.
        # Build both custom configs from the verified snapshot, then load the
        # published ``language_model.*`` state dict directly. Calling the
        # generic AutoModel loader here would prepend ``model.`` and silently
        # leave the language model randomly initialized on current Transformers.
        snapshot = Path(snapshot_path).resolve()
        processor_snapshot = snapshot / "processor"
        if not (processor_snapshot / "configuration_uniformer.py").is_file():
            raise RuntimeError(
                "CXRMate-ED verified UniFormer processor code is missing"
            )
        cache_dir = Path(HF_MODULES_CACHE).resolve()
        if not is_within_allowed_roots(cache_dir):
            raise RuntimeError(
                "CXRMate-ED remote-code cache is outside application storage"
            )
        dynamic_options = {
            "cache_dir": str(cache_dir),
            "local_files_only": True,
        }
        config_class = get_class_from_dynamic_module(
            "configuration_cxrmate_ed.CXRMateEDConfig",
            str(snapshot),
            **dynamic_options,
        )
        uniformer_config_class = get_class_from_dynamic_module(
            "configuration_uniformer.UniFormerWithProjectionHeadConfig",
            str(processor_snapshot),
            **dynamic_options,
        )
        model_class = get_class_from_dynamic_module(
            "modelling_cxrmate_ed.CXRMateEDModel",
            str(snapshot),
            **dynamic_options,
        )
        raw_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        vision_values = dict(raw_config.get("vision_config") or {})
        vision_values["_name_or_path"] = str(processor_snapshot)
        vision_values["auto_map"] = {
            "AutoConfig": "configuration_uniformer.UniFormerWithProjectionHeadConfig",
            "AutoModel": "modelling_uniformer.UniFormerModel",
        }
        vision_config = uniformer_config_class(**vision_values)
        raw_config["vision_config"] = vision_config
        config = config_class.from_dict(raw_config)
        model = model_class(config)
        weights = load_file(str(snapshot / "model.safetensors"), device="cpu")
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing or unexpected:
            missing_preview = ", ".join(str(key) for key in missing[:4])
            unexpected_preview = ", ".join(str(key) for key in unexpected[:4])
            raise RuntimeError(
                "CXRMate-ED checkpoint keys do not match its model architecture "
                f"(missing: {missing_preview or 'none'}; "
                f"unexpected: {unexpected_preview or 'none'})."
            )
        dtype = load_options.get("torch_dtype")
        if isinstance(dtype, torch.dtype) and dtype != torch.float32:
            model.to(dtype=dtype)
        device_map = load_options.get("device_map")
        if device_map == "cuda" or (device_map == "auto" and torch.cuda.is_available()):
            model.to("cuda")
        return model

    # -------------------------------------------------------------------------
    def load_processor(
        self,
        snapshot_path: str,
        *,
        processor_loader: str,
        load_options: dict[str, Any],
    ) -> Any:
        return AutoTokenizer.from_pretrained(snapshot_path, **load_options)

    # -------------------------------------------------------------------------
    def generate_study(
        self,
        *,
        model: Any,
        processor: Any,
        images: list[StudyImage],
        profile: GenerationProfile,
        clinical_context: str,
        move_inputs: MoveInputs,
        stopping_criteria: StoppingCriteriaValue,
        output_sections: list[str],
    ) -> StudyGeneration:
        image_tensors: list[torch.Tensor] = []
        for item in images:
            transformed = model.test_transforms(pil_to_tensor(item.image))
            image_tensors.append(cast(torch.Tensor, torch.as_tensor(transformed)))
        batch = torch.stack(image_tensors, dim=0).unsqueeze(0)
        batch = move_inputs({"images": batch}, model)["images"]
        context = clinical_context.strip() or None
        inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = (
            model.prepare_inputs(
                tokenizer=processor,
                images=batch,
                image_time_deltas=[[model.zero_time_delta_value] * len(images)],
                study_id=[0],
                indication=[[context]],
                history=[[None]],
            )
        )
        moved = move_inputs(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "bos_token_ids": bos_token_ids,
            },
            model,
        )
        _ensure_cxrmate_ed_cache_compatibility(model)
        generation_kwargs = self.generation_profiles[profile]
        output = model.generate(
            input_ids=moved["bos_token_ids"],
            decoder_inputs_embeds=moved["inputs_embeds"],
            decoder_token_type_ids=moved["token_type_ids"],
            prompt_attention_mask=moved["attention_mask"],
            prompt_position_ids=moved["position_ids"],
            special_token_ids=[processor.sep_token_id],
            **generation_kwargs,
            return_dict_in_generate=True,
            stopping_criteria=stopping_criteria,
        )["sequences"]
        findings, impression = model.split_and_decode_sections(
            output,
            [processor.sep_token_id, processor.eos_token_id],
            processor,
        )
        report = f"Findings:\n{findings[0].strip()}\n\nImpression:\n{impression[0].strip()}".strip()
        return StudyGeneration(
            report=report,
            display_sections={
                "findings": findings[0].strip(),
                "impression": impression[0].strip(),
            },
            metadata=[self._metadata(item, batch) for item in images],
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _metadata(item: StudyImage, tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "filename": item.stored.filename,
            "original_dimensions": {
                "width": item.original_dimensions[0],
                "height": item.original_dimensions[1],
            },
            "processed_tensor_dimensions": [int(value) for value in tensor.shape],
            "processor_loader": "auto_tokenizer",
            "input_scope": "study",
        }


###############################################################################
class CXRMate2Adapter(StandardImageTextAdapter):
    """Published CXRMate-2 processor and findings/impression decoder."""

    supports_study = True

    # -------------------------------------------------------------------------
    def generate_study(
        self,
        *,
        model: Any,
        processor: Any,
        images: list[StudyImage],
        profile: GenerationProfile,
        clinical_context: str,
        move_inputs: MoveInputs,
        stopping_criteria: StoppingCriteriaValue,
        output_sections: list[str],
    ) -> StudyGeneration:
        del profile
        processed = processor(
            images=[item.image for item in images],
            indication=clinical_context.strip() or None,
        )
        processed = move_inputs(processed, model)
        generated_ids = model.generate(
            **processed,
            max_length=256,
            num_beams=4,
            stopping_criteria=stopping_criteria,
        )
        findings, impression = processor.split_and_decode_sections(generated_ids)
        report = f"Findings:\n{findings[0].strip()}\n\nImpression:\n{impression[0].strip()}".strip()
        return StudyGeneration(
            report=report,
            display_sections={
                "findings": findings[0].strip(),
                "impression": impression[0].strip(),
            },
            metadata=[self._metadata(item, processed) for item in images],
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _metadata(item: StudyImage, inputs: Any) -> dict[str, Any]:
        dimensions = StandardImageTextAdapter.processed_dimensions(inputs)
        return {
            "filename": item.stored.filename,
            "original_dimensions": {
                "width": item.original_dimensions[0],
                "height": item.original_dimensions[1],
            },
            "processed_tensor_dimensions": dimensions,
            "processor_loader": "auto",
            "input_scope": "study",
        }


###############################################################################
class MedGemmaAdapter(ChatVisionStudyAdapter):
    """MedGemma's standard processor with a research-only report prompt."""

    # -------------------------------------------------------------------------
    @staticmethod
    def prompt(profile: GenerationProfile, clinical_context: str) -> str:
        return (
            "Create a concise radiology report draft from this medical imaging study. "
            "Return only the report text, with Findings and Impression headings when appropriate. "
            "Do not provide chain-of-thought or diagnostic certainty. "
            f"{StandardImageTextAdapter.prompt(profile, clinical_context)}"
        )


ADAPTERS: dict[str, type[StandardImageTextAdapter]] = {
    "medgemma": MedGemmaAdapter,
    "chexone": CheXOneAdapter,
    "cxrmate_multi": CXRMateMultiAdapter,
    "cxrmate_ed": CXRMateEDAdapter,
    "cxrmate2": CXRMate2Adapter,
}
