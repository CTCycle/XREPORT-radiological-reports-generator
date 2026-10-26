from __future__ import annotations

import sys
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.api.system import get_router


###############################################################################
def build_client() -> TestClient:
    app = FastAPI()
    app.include_router(get_router(), prefix="/api")
    return TestClient(app)


###############################################################################
def test_system_status_reports_ram_and_cuda_unavailable(monkeypatch) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    payload = build_client().get("/api/system/status").json()

    assert payload["cuda"] == {"available": False, "device_name": None}
    assert payload["vram"] is None
    assert 0 < payload["ram"]["used_gb"] <= payload["ram"]["total_gb"]


###############################################################################
def test_system_status_reports_vram_when_cuda_available(monkeypatch) -> None:
    gb = 1024**3
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: "Test GPU",
        mem_get_info=lambda index: (6 * gb, 8 * gb),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    payload = build_client().get("/api/system/status").json()

    assert payload["cuda"] == {"available": True, "device_name": "Test GPU"}
    assert payload["vram"] == {"used_gb": 2.0, "total_gb": 8.0}
