from __future__ import annotations

from PIL import Image
import pytest
import torch

from server.domain.inference import InferenceImage
from server.models.inference.providers import adapters as adapters_module
from server.models.inference.providers.adapters import (
    CheXOneAdapter,
    CXRMate2Adapter,
    CXRMateEDAdapter,
    CXRMateMultiAdapter,
    MedGemmaAdapter,
    StudyImage,
    _ensure_cxrmate2_generation_dtype,
)

###############################################################################
@pytest.mark.parametrize(
    ("profile", "max_length", "num_beams"),
    (
        ("deterministic", 256, 1),
        ("concise", 160, 1),
        ("detailed", 384, 4),
    ),
)
def test_cxrmate_ed_passes_clinical_context_and_all_study_images(
    profile: str, max_length: int, num_beams: int
) -> None:

    ###############################################################################
    class ModelStub:
        zero_time_delta_value = 0

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.prepare_call: dict[str, object] = {}
            self.generate_call: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def test_transforms(self, image_tensor):
            return image_tensor.float()

        # -------------------------------------------------------------------------
        def prepare_inputs(self, **kwargs):
            self.prepare_call = kwargs
            return (
                torch.zeros((1, 2, 4)),
                torch.ones((1, 2), dtype=torch.long),
                torch.ones((1, 2), dtype=torch.long),
                torch.zeros((1, 2), dtype=torch.long),
                torch.ones((1, 1), dtype=torch.long),
            )

        # -------------------------------------------------------------------------
        def generate(self, **kwargs):
            self.generate_call = kwargs
            return {"sequences": torch.tensor([[1, 2]])}

        # -------------------------------------------------------------------------
        def split_and_decode_sections(self, output, token_ids, processor):
            return ["finding"], ["impression"]

    ###############################################################################
    class ProcessorStub:
        sep_token_id = 10
        eos_token_id = 11

    model = ModelStub()
    images = [
        StudyImage(
            stored=InferenceImage("a.png", "image/png", b"a", 1),
            image=Image.new("RGB", (4, 4), color=(0, 0, 0)),
            original_dimensions=(4, 4),
        ),
        StudyImage(
            stored=InferenceImage("b.png", "image/png", b"b", 1),
            image=Image.new("RGB", (4, 4), color=(255, 255, 255)),
            original_dimensions=(4, 4),
        ),
    ]

    stopping_criteria = object()
    result = CXRMateEDAdapter().generate_study(
        model=model,
        processor=ProcessorStub(),
        images=images,
        profile=profile,  # type: ignore[arg-type]
        clinical_context="shortness of breath",
        move_inputs=lambda inputs, _model: inputs,
        stopping_criteria=stopping_criteria,
        output_sections=["findings", "impression"],
    )

    assert model.prepare_call["indication"] == [["shortness of breath"]]
    assert model.prepare_call["images"].shape == (1, 2, 3, 4, 4)
    assert not torch.equal(
        model.prepare_call["images"][0, 0], model.prepare_call["images"][0, 1]
    )
    assert model.generate_call["max_length"] == max_length
    assert model.generate_call["num_beams"] == num_beams
    assert model.generate_call["do_sample"] is False
    assert model.generate_call["stopping_criteria"] is stopping_criteria
    assert result.display_sections == {
        "findings": "finding",
        "impression": "impression",
    }


###############################################################################
def _study_images() -> list[StudyImage]:
    return [
        StudyImage(
            stored=InferenceImage("a.png", "image/png", b"a", 1),
            image=Image.new("RGB", (4, 4), color=(0, 0, 0)),
            original_dimensions=(4, 4),
        ),
        StudyImage(
            stored=InferenceImage("b.png", "image/png", b"b", 1),
            image=Image.new("RGB", (4, 4), color=(255, 255, 255)),
            original_dimensions=(4, 4),
        ),
    ]


###############################################################################
@pytest.mark.parametrize(
    ("profile", "max_length", "num_beams"),
    (("deterministic", 256, 1), ("concise", 160, 1), ("detailed", 256, 4)),
)
def test_cxrmate_multi_uses_all_images_profile_and_stopping_criteria(
    profile: str, max_length: int, num_beams: int
) -> None:

    ###############################################################################
    class ImageProcessorStub:
        size = {"shortest_edge": 4}
        image_mean = (0.5, 0.5, 0.5)
        image_std = (0.5, 0.5, 0.5)

    ###############################################################################
    class TokenizerStub:
        sep_token_id = 10
        bos_token_id = 11
        eos_token_id = 12
        pad_token_id = 13

    ###############################################################################
    class ModelStub:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.generate_call: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def generate(self, **kwargs: object):
            self.generate_call = kwargs
            return type("Generation", (), {"sequences": torch.tensor([[1, 2]])})()

        # -------------------------------------------------------------------------
        def split_and_decode_sections(self, _output, _token_ids, _tokenizer):
            return ["finding"], ["impression"]

    model = ModelStub()
    stopping_criteria = object()
    result = CXRMateMultiAdapter().generate_study(
        model=model,
        processor={
            "image_processor": ImageProcessorStub(),
            "tokenizer": TokenizerStub(),
        },
        images=_study_images(),
        profile=profile,  # type: ignore[arg-type]
        clinical_context="",
        move_inputs=lambda inputs, _model: inputs,
        stopping_criteria=stopping_criteria,
        output_sections=["findings", "impression"],
    )

    pixel_values = model.generate_call["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (1, 2, 3, 4, 4)
    assert not torch.equal(pixel_values[0, 0], pixel_values[0, 1])
    assert model.generate_call["max_length"] == max_length
    assert model.generate_call["num_beams"] == num_beams
    assert model.generate_call["do_sample"] is False
    assert model.generate_call["stopping_criteria"] is stopping_criteria
    assert result.display_sections == {
        "findings": "finding",
        "impression": "impression",
    }


###############################################################################
@pytest.mark.parametrize(
    ("profile", "max_length", "num_beams"),
    (("deterministic", 256, 1), ("concise", 160, 1), ("detailed", 256, 4)),
)
def test_cxrmate2_forwards_study_and_profile_to_published_processor(
    profile: str, max_length: int, num_beams: int
) -> None:

    ###############################################################################
    class ProcessorStub:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.process_call: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def __call__(self, **kwargs: object):
            self.process_call = kwargs
            return {
                "pixel_values": torch.zeros((1, 2, 3, 4, 4)),
                "input_ids": torch.ones((1, 2), dtype=torch.long),
            }

        # -------------------------------------------------------------------------
        def split_and_decode_sections(self, _generated_ids):
            return ["finding"], ["impression"]

    ###############################################################################
    class ModelStub:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.generate_call: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def generate(self, **kwargs: object):
            self.generate_call = kwargs
            return torch.tensor([[1, 2]])

    processor = ProcessorStub()
    model = ModelStub()
    stopping_criteria = object()
    result = CXRMate2Adapter().generate_study(
        model=model,
        processor=processor,
        images=_study_images(),
        profile=profile,  # type: ignore[arg-type]
        clinical_context="portable chest",
        move_inputs=lambda inputs, _model: inputs,
        stopping_criteria=stopping_criteria,
        output_sections=["findings", "impression"],
    )

    assert processor.process_call["images"] == [[item.image for item in _study_images()]]
    assert processor.process_call["indication"] == "portable chest"
    assert model.generate_call["max_length"] == max_length
    assert model.generate_call["num_beams"] == num_beams
    assert model.generate_call["do_sample"] is False
    assert model.generate_call["stopping_criteria"] is stopping_criteria
    assert result.display_sections == {
        "findings": "finding",
        "impression": "impression",
    }


###############################################################################
def test_cxrmate2_casts_generation_time_deltas_to_model_dtype() -> None:

    ###############################################################################
    class ModelStub:
        device = torch.device("cpu")
        dtype = torch.bfloat16

        # -------------------------------------------------------------------------
        def prepare_inputs_for_generation(self, *_args, **_kwargs):
            return {"time_deltas": torch.zeros((1, 2), dtype=torch.float32)}

    model = ModelStub()
    _ensure_cxrmate2_generation_dtype(
        model,
        lambda inputs, _model: {
            key: value.to(dtype=torch.bfloat16)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
            for key, value in inputs.items()
        },
    )

    prepared = model.prepare_inputs_for_generation(torch.ones((1, 1), dtype=torch.long))

    assert prepared["time_deltas"].dtype is torch.bfloat16


###############################################################################
def test_cxrmate_ed_generation_profiles_are_model_specific() -> None:
    assert CXRMateEDAdapter.generation_profiles == {
        "deterministic": {"max_length": 256, "num_beams": 1, "do_sample": False},
        "concise": {"max_length": 160, "num_beams": 1, "do_sample": False},
        "detailed": {"max_length": 384, "num_beams": 4, "do_sample": False},
    }


###############################################################################
@pytest.mark.parametrize(
    ("profile", "max_new_tokens"),
    (
        ("deterministic", 768),
        ("concise", 384),
        ("detailed", 1024),
    ),
)
def test_chexone_uses_published_multi_image_vision_path_and_strips_prompt(
    monkeypatch, profile: str, max_new_tokens: int
) -> None:
    images = _study_images()
    vision_call: dict[str, object] = {}

    def process_vision_info(messages):
        vision_call["messages"] = messages
        return ([item.image for item in images], None)

    monkeypatch.setattr(adapters_module, "process_vision_info", process_vision_info)

    ###############################################################################
    class ProcessorStub:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.template_messages = None
            self.processor_call: dict[str, object] = {}
            self.decoded_ids = None

        # -------------------------------------------------------------------------
        def apply_chat_template(self, messages, **_kwargs: object) -> str:
            self.template_messages = messages
            return "rendered chat prompt"

        # -------------------------------------------------------------------------
        def __call__(self, **kwargs: object):
            self.processor_call = kwargs

            ###############################################################################
            class VisionInputs(dict[str, torch.Tensor]):
                input_ids = torch.tensor([[1, 2]])

                # -------------------------------------------------------------------------
                def __init__(self) -> None:
                    super().__init__(
                        input_ids=self.input_ids,
                        pixel_values=torch.zeros((1, 2, 3, 4, 4)),
                    )

            return VisionInputs()

        # -------------------------------------------------------------------------
        def batch_decode(self, generated_ids, **_kwargs: object) -> list[str]:
            self.decoded_ids = generated_ids
            return ["Findings: clear\nImpression: normal"]

    ###############################################################################
    class ModelStub:

        # -------------------------------------------------------------------------
        def generate(self, **kwargs: object):
            self.generate_call = kwargs
            return [torch.tensor([1, 2, 3, 4])]

    processor = ProcessorStub()
    model = ModelStub()
    stopping_criteria = object()
    result = CheXOneAdapter().generate_study(
        model=model,
        processor=processor,
        images=images,
        profile=profile,  # type: ignore[arg-type]
        clinical_context="cough",
        move_inputs=lambda inputs, _model: inputs,
        stopping_criteria=stopping_criteria,
        output_sections=["findings", "impression"],
    )

    assert vision_call["messages"] == processor.template_messages
    assert processor.processor_call["images"] == [item.image for item in images]
    prompt = processor.template_messages[0]["content"][-1]["text"]
    assert "final Findings and Impression" in prompt
    assert "reasoning traces" in prompt
    assert model.generate_call["max_new_tokens"] == max_new_tokens
    assert model.generate_call["stopping_criteria"] is stopping_criteria
    assert torch.equal(processor.decoded_ids[0], torch.tensor([3, 4]))
    assert result.display_sections == {
        "findings": "clear",
        "impression": "normal",
    }


###############################################################################
@pytest.mark.parametrize(
    ("profile", "max_new_tokens"),
    (("deterministic", 768), ("concise", 384), ("detailed", 1536)),
)
def test_medgemma_uses_multi_image_context_raw_report_contract_and_profile(
    profile: str, max_new_tokens: int
) -> None:
    images = _study_images()

    ###############################################################################
    class ProcessorStub:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.messages = None

        # -------------------------------------------------------------------------
        def apply_chat_template(self, messages, **_kwargs: object):
            self.messages = messages
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": torch.zeros((1, 2, 3, 4, 4)),
            }

        # -------------------------------------------------------------------------
        def decode(self, generated, **_kwargs: object) -> str:
            assert torch.equal(generated, torch.tensor([3, 4]))
            return "raw report"

    ###############################################################################
    class ModelStub:

        # -------------------------------------------------------------------------
        def generate(self, **kwargs: object):
            self.generate_call = kwargs
            return [torch.tensor([1, 2, 3, 4])]

    processor = ProcessorStub()
    model = ModelStub()
    stopping_criteria = object()
    result = MedGemmaAdapter().generate_study(
        model=model,
        processor=processor,
        images=images,
        profile=profile,  # type: ignore[arg-type]
        clinical_context="cough",
        move_inputs=lambda inputs, _model: inputs,
        stopping_criteria=stopping_criteria,
        output_sections=["raw_report"],
    )

    assert len(processor.messages[0]["content"]) == 3
    assert processor.messages[0]["content"][-1]["text"].find("cough") >= 0
    assert model.generate_call["max_new_tokens"] == max_new_tokens
    assert model.generate_call["do_sample"] is False
    assert model.generate_call["stopping_criteria"] is stopping_criteria
    assert result.report == "raw report"
    assert result.display_sections == {"raw_report": "raw report"}
