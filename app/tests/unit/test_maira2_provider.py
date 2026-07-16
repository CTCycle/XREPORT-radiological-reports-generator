from __future__ import annotations

import json

import httpx
import pytest

from server.configurations import InferenceSettings
from server.domain.inference import InferenceImage
from server.models.inference.providers.maira2 import Maira2Provider


REVISION = "a" * 40


def _settings(**overrides: object) -> InferenceSettings:
    values = {
        "ollama_base_url": "http://127.0.0.1:11434",
        "ollama_keep_alive": "5m",
        "hf_local_only": True,
        "hf_cache_dir": None,
        "hf_medgemma_revision": None,
        "device": "auto",
        "max_loaded_models": 1,
        "model_timeout": 600,
        "maira2_enabled": True,
        "maira2_worker_url": "http://127.0.0.1:5010",
        "maira2_revision": REVISION,
    }
    values.update(overrides)
    return InferenceSettings(**values)  # type: ignore[arg-type]


def _response(method: str, url: str, payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request(method, url),
    )


def test_provider_rejects_non_loopback_worker_url() -> None:
    with pytest.raises(ValueError, match="loopback"):
        Maira2Provider(_settings(maira2_worker_url="http://192.168.1.20:5010"))


def test_availability_requires_matching_pinned_worker(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: _response(
            "GET",
            str(args[0]),
            {"status": "ready", "model": "microsoft/maira-2", "revision": REVISION},
        ),
    )

    assert Maira2Provider(_settings()).availability() == ("ready", None)
    assert (
        Maira2Provider(_settings(maira2_revision="main")).availability()[0]
        == "incompatible"
    )


def test_generation_sends_one_base64_image_and_returns_findings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        Maira2Provider,
        "availability",
        lambda self: ("ready", None),
    )

    def post(
        url: str, *, json: dict[str, object], timeout: httpx.Timeout
    ) -> httpx.Response:
        captured.update(json)
        return _response("POST", url, {"findings": "No focal airspace opacity."})

    monkeypatch.setattr(httpx, "post", post)
    progress: list[tuple[int, int, dict[str, str]]] = []
    reports = Maira2Provider(_settings()).generate(
        repository_id="microsoft/maira-2",
        revision=REVISION,
        profile="deterministic",
        clinical_context="Dyspnea",
        images=[InferenceImage("cxr.png", "image/png", b"image", 5)],
        should_stop=lambda: False,
        report_progress=lambda current, total, result: progress.append(
            (current, total, result)
        ),
    )

    assert json.loads(json.dumps(captured))["image"] == "aW1hZ2U="
    assert reports == {"cxr.png": "Findings\nNo focal airspace opacity."}
    assert progress == [(1, 1, reports)]
