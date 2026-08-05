from __future__ import annotations

from unittest.mock import MagicMock

import torch
from PIL import Image

from server.models.inference.providers.adapters import BlipCxrAdapter
from server.models.inference.providers.huggingface import HuggingFaceProvider


###############################################################################
def test_blip_uses_indication_prefix_and_full_sequence_decode() -> None:
    adapter = BlipCxrAdapter()
    processor = MagicMock()
    processor.return_value = {
        "input_ids": torch.tensor([[10, 11]]),
        "pixel_values": torch.zeros((1, 3, 224, 224)),
    }
    image = Image.new("RGB", (4, 3), "white")

    prompt = adapter.prompt("detailed", "Pneumonia")
    inputs, input_length = adapter.build_inputs(processor, image, prompt)

    assert prompt == "indication:Pneumonia"
    assert input_length == 2
    processor.assert_called_once_with(
        images=image,
        text=prompt,
        return_tensors="pt",
    )
    processor.decode.return_value = "  Report text  "
    assert adapter.decode(processor, torch.tensor([[1, 2, 3]]), input_length=2) == "  Report text  "
    assert adapter.generation_kwargs("detailed")["max_length"] == 512


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
