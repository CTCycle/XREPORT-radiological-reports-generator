from __future__ import annotations

import base64

import httpx
import pytest

from server.configurations import InferenceSettings
from server.domain.inference import InferenceImage
from server.models.inference.providers.ollama import OllamaProvider

###############################################################################
def _settings() -> InferenceSettings:
    return InferenceSettings(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_keep_alive="7m",
        hf_local_only=True,
        hf_cache_dir=None,
        hf_medgemma_revision=None,
        device="auto",
        max_loaded_models=1,
        model_timeout=30,
    )

###############################################################################
def _response(url: str, payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

###############################################################################
def test_generate_uses_chat_with_base64_image_and_keep_alive(monkeypatch) -> None:
    captured: dict[str, object] = {}
    tags_url = "http://127.0.0.1:11434/api/tags"
    chat_url = "http://127.0.0.1:11434/api/chat"
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: _response(tags_url, {"models": [{"name": "medgemma:4b"}]}),
    )

    def post(url: str, **kwargs: object) -> httpx.Response:
        captured.update(url=url, **kwargs)
        return _response(chat_url, {"message": {"content": "Findings: clear."}})

    monkeypatch.setattr(httpx, "post", post)
    progress: list[tuple[int, int, dict[str, str]]] = []

    reports = OllamaProvider(_settings()).generate(
        model="medgemma:4b",
        profile="concise",
        clinical_context="Cough",
        images=[InferenceImage(filename="scan.png", content_type="image/png", data=b"image", size_bytes=5)],
        should_stop=lambda: False,
        report_progress=lambda current, total, values: progress.append((current, total, values)),
    )

    assert captured["url"] == chat_url
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["keep_alive"] == "7m"
    assert payload["stream"] is False
    assert payload["messages"][0]["images"] == [base64.b64encode(b"image").decode("ascii")]
    assert reports == {"scan.png": "Findings: clear."}
    assert progress == [(1, 1, reports)]

###############################################################################
def test_generate_reports_runtime_unavailable(monkeypatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", unavailable)

    with pytest.raises(RuntimeError, match="runtime is unavailable"):
        OllamaProvider(_settings()).generate(
            model="medgemma:4b",
            profile="deterministic",
            clinical_context="",
            images=[InferenceImage(filename="scan.png", content_type="image/png", data=b"x", size_bytes=1)],
            should_stop=lambda: False,
            report_progress=lambda *_: None,
        )

###############################################################################
def test_generate_reports_model_not_installed(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: _response("http://127.0.0.1:11434/api/tags", {"models": []}),
    )

    with pytest.raises(RuntimeError, match="not installed"):
        OllamaProvider(_settings()).generate(
            model="medgemma:4b",
            profile="deterministic",
            clinical_context="",
            images=[InferenceImage(filename="scan.png", content_type="image/png", data=b"x", size_bytes=1)],
            should_stop=lambda: False,
            report_progress=lambda *_: None,
        )
