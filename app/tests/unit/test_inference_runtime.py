from __future__ import annotations

from types import SimpleNamespace

import torch

from server.services.inference_runtime import InferenceRuntimeCoordinator


###############################################################################
def test_xreport_runtime_metadata_reports_requested_and_resolved_devices(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = SimpleNamespace(
        variables=[
            SimpleNamespace(value=SimpleNamespace(device="cuda:0")),
            SimpleNamespace(value=SimpleNamespace(device="cuda:0")),
        ]
    )

    metadata = InferenceRuntimeCoordinator._xreport_runtime_metadata(model, "auto")

    assert metadata == {
        "requested_device": "auto",
        "resolved_device": "cuda:0",
        "resolved_devices": ["cuda:0"],
        "cuda_available": True,
        "cuda_used": True,
    }
