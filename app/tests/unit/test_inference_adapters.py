from __future__ import annotations

from unittest.mock import MagicMock

import torch

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
