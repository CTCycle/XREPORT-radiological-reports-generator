from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
import torch

from server.configurations import InferenceSettings
from server.domain.inference import InferenceImage
from server.models.inference.providers.huggingface import HuggingFaceProvider


REVISION = "a" * 40

###############################################################################
def _settings(cache_dir: Path) -> InferenceSettings:
    return InferenceSettings(
        hf_local_only=True,
        hf_cache_dir=str(cache_dir),
        device="cpu",
        max_loaded_models=1,
        model_timeout=600,
    )

###############################################################################
def _png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (3, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()

###############################################################################
class Inputs(dict[str, torch.Tensor]):

    # -------------------------------------------------------------------------
    def to(self, *_args: object, **_kwargs: object) -> "Inputs":
        return self

###############################################################################
def _manifest() -> dict[str, object]:
    return {
        "revision": REVISION,
        "model_loader": "image_text_to_text",
        "processor_loader": "auto",
        "adapter": "medgemma",
        "trust_remote_code": False,
        "remote_code_approved": False,
        "max_current_images": 1,
        "preferred_dtype": "float32",
    }

###############################################################################
def test_generate_uses_manifest_loaders_revision_and_records_dimensions(monkeypatch) -> None:
    cache_path = Path("assets/QA/test-huggingface-cache")
    calls: dict[str, object] = {}
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    processor = MagicMock()
    processor.apply_chat_template.return_value = Inputs({
        "input_ids": torch.tensor([[1, 2]]),
        "pixel_values": torch.zeros((1, 3, 896, 896)),
    })
    processor.decode.return_value = "Findings: no acute abnormality."

    def snapshot_download(**kwargs: object) -> str:
        calls["snapshot"] = kwargs
        return str(cache_path / "snapshot")

    def load_processor(path: str, **kwargs: object) -> MagicMock:
        calls["processor"] = (path, kwargs)
        return processor

    def load_model(path: str, **kwargs: object) -> MagicMock:
        calls["model"] = (path, kwargs)
        return model

    monkeypatch.setattr(
        "server.models.inference.providers.huggingface.snapshot_download",
        snapshot_download,
    )
    monkeypatch.setattr(
        "server.models.inference.providers.adapters.AutoProcessor.from_pretrained",
        load_processor,
    )
    monkeypatch.setattr(
        "server.models.inference.providers.adapters.AutoModelForImageTextToText.from_pretrained",
        load_model,
    )
    progress: list[tuple[object, ...]] = []

    reports = HuggingFaceProvider(_settings(cache_path)).generate(
        repository_id="google/medgemma-1.5-4b-it",
        manifest=_manifest(),
        profile="deterministic",
        clinical_context="Cough",
        images=[InferenceImage(filename="scan.png", content_type="image/png", data=_png(), size_bytes=69)],
        should_stop=lambda: False,
        report_progress=lambda *values: progress.append(values),
    )

    assert calls["snapshot"] == {
        "repo_id": "google/medgemma-1.5-4b-it",
        "revision": REVISION,
        "cache_dir": str(cache_path),
        "local_files_only": True,
    }
    processor_call = calls["processor"]
    model_call = calls["model"]
    assert isinstance(processor_call, tuple)
    assert isinstance(model_call, tuple)
    for _, options in (processor_call, model_call):
        assert isinstance(options, dict)
        assert options["local_files_only"] is True
        assert options["trust_remote_code"] is False
    assert model.generate.call_args.kwargs["do_sample"] is False
    assert reports == {"scan.png": "Findings: no acute abnormality."}
    assert progress[0][3] == [{
        "filename": "scan.png",
        "original_dimensions": {"width": 3, "height": 2},
        "processed_tensor_dimensions": [1, 3, 896, 896],
        "processor_loader": "auto",
        "model_loader": "image_text_to_text",
        "adapter": "medgemma",
    }]

###############################################################################
def test_provider_rejects_unpinned_revision() -> None:
    manifest = _manifest()
    manifest["revision"] = "main"
    image = InferenceImage(filename="scan.png", content_type="image/png", data=_png(), size_bytes=69)

    try:
        HuggingFaceProvider(_settings(Path("assets/QA/test-huggingface-cache"))).generate(
            repository_id="google/medgemma-1.5-4b-it",
            manifest=manifest,
            profile="deterministic",
            clinical_context="",
            images=[image],
            should_stop=lambda: False,
            report_progress=lambda *_: None,
        )
    except RuntimeError as exc:
        assert "pinned 40-character revision" in str(exc)
    else:
        raise AssertionError("Unpinned revision was accepted")

###############################################################################
def test_provider_rejects_multiple_images() -> None:
    image = InferenceImage(filename="scan.png", content_type="image/png", data=_png(), size_bytes=69)

    try:
        HuggingFaceProvider(_settings(Path("assets/QA/test-huggingface-cache"))).generate(
            repository_id="google/medgemma-1.5-4b-it",
            manifest=_manifest(),
            profile="detailed",
            clinical_context="",
            images=[image, image],
            should_stop=lambda: False,
            report_progress=lambda *_: None,
        )
    except ValueError as exc:
        assert "at most 1" in str(exc)
    else:
        raise AssertionError("Multiple images were accepted")
