from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image
import torch

from server.domain.inference import InferenceImage
from server.models.inference.providers.adapters import CXRMateEDAdapter, StudyImage
from server.models.inference.providers.huggingface import HuggingFaceProvider

###############################################################################
def test_move_inputs_casts_float_tensors_without_changing_token_dtype() -> None:
    model = MagicMock()
    model.device = torch.device("cpu")
    model.dtype = torch.bfloat16
    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
        "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
    }

    moved = HuggingFaceProvider._move_inputs(inputs, model)

    assert moved["input_ids"].dtype == torch.int64
    assert moved["pixel_values"].dtype == torch.bfloat16

###############################################################################
def test_cxrmate_ed_passes_profile_context_and_image_inputs() -> None:
    class ModelStub:
        zero_time_delta_value = 0

        def __init__(self) -> None:
            self.prepare_call: dict[str, object] = {}
            self.generate_call: dict[str, object] = {}

        def test_transforms(self, image_tensor):
            return image_tensor.float()

        def prepare_inputs(self, **kwargs):
            self.prepare_call = kwargs
            return (
                torch.zeros((1, 2, 4)),
                torch.ones((1, 2), dtype=torch.long),
                torch.ones((1, 2), dtype=torch.long),
                torch.zeros((1, 2), dtype=torch.long),
                torch.ones((1, 1), dtype=torch.long),
            )

        def generate(self, **kwargs):
            self.generate_call = kwargs
            return {"sequences": torch.tensor([[1, 2]])}

        def split_and_decode_sections(self, output, token_ids, processor):
            return ["finding"], ["impression"]

    class ProcessorStub:
        sep_token_id = 10
        eos_token_id = 11

    model = ModelStub()
    adapter = CXRMateEDAdapter()
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

    result = adapter.generate_study(
        model=model,
        processor=ProcessorStub(),
        images=images,
        profile="concise",
        clinical_context="shortness of breath",
        move_inputs=lambda inputs, _model: inputs,
        stopping_criteria=None,
        output_sections=["findings", "impression"],
    )

    assert model.prepare_call["indication"] == [["shortness of breath"]]
    assert model.prepare_call["images"].shape == (1, 2, 3, 4, 4)
    assert not torch.equal(model.prepare_call["images"][0, 0], model.prepare_call["images"][0, 1])
    assert model.generate_call["max_length"] == 160
    assert model.generate_call["num_beams"] == 1
    assert model.generate_call["do_sample"] is False
    assert result.display_sections == {"findings": "finding", "impression": "impression"}

###############################################################################
def test_huggingface_provenance_records_degraded_quality_warning() -> None:
    provenance = HuggingFaceProvider._provenance(
        {
            "repository_id": "aehrc/cxrmate-ed",
            "revision": "a" * 40,
            "adapter": "cxrmate_ed",
            "model_loader": "causal_lm",
            "processor_loader": "auto",
            "validation_status": "degraded",
            "validation_message": "Research-only warning",
        },
        "deterministic",
        "",
    )

    assert provenance["quality_warnings"] == ["sensitivity_canary_failed"]
    assert provenance["validation_status"] == "degraded"
