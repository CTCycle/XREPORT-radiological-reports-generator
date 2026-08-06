from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

from PIL import Image
import torch

from server.configurations import InferenceSettings
from server.domain.inference import InferenceImage
from server.models.inference.providers.adapters import MedGemmaAdapter
from server.models.inference.providers.huggingface import HuggingFaceProvider


REVISION = "a" * 40

###############################################################################
def _settings() -> InferenceSettings:
    return InferenceSettings(
        hf_local_only=True,
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
def _patch_runtime(monkeypatch, model: MagicMock, processor: MagicMock) -> None:
    monkeypatch.setattr(
        HuggingFaceProvider,
        "_load",
        lambda _self, _manifest: (model, processor, MedGemmaAdapter()),
    )
    monkeypatch.setattr(
        "server.models.inference.providers.adapters.AutoProcessor.from_pretrained",
        lambda _path, **_kwargs: processor,
    )
    monkeypatch.setattr(
        "server.models.inference.providers.adapters.AutoModelForImageTextToText.from_pretrained",
        lambda _path, **_kwargs: model,
    )

###############################################################################
def _processor_inputs() -> Inputs:
    return Inputs({
        "input_ids": torch.tensor([[1, 2]]),
        "pixel_values": torch.zeros((1, 3, 8, 9)),
    })

###############################################################################
def test_generate_uses_manifest_loaders_revision_and_records_dimensions(monkeypatch, tmp_path) -> None:
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
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

    def load_processor(path: str, **kwargs: object) -> MagicMock:
        calls["processor"] = (path, kwargs)
        return processor

    def load_model(path: str, **kwargs: object) -> MagicMock:
        calls["model"] = (path, kwargs)
        return model

    monkeypatch.setattr(
        "server.models.inference.providers.adapters.AutoProcessor.from_pretrained",
        load_processor,
    )
    monkeypatch.setattr(
        "server.models.inference.providers.adapters.AutoModelForImageTextToText.from_pretrained",
        load_model,
    )
    progress: list[tuple[object, ...]] = []

    result = HuggingFaceProvider(_settings()).generate(
        repository_id="google/medgemma-1.5-4b-it",
        manifest={**_manifest(), "local_snapshot_path": str(snapshot_path.resolve())},
        profile="deterministic",
        clinical_context="Cough",
        images=[InferenceImage(filename="scan.png", content_type="image/png", data=_png(), size_bytes=69)],
        should_stop=lambda: False,
        report_progress=lambda *values: progress.append(values),
    )

    processor_call = calls["processor"]
    model_call = calls["model"]
    assert isinstance(processor_call, tuple)
    assert isinstance(model_call, tuple)
    for _, options in (processor_call, model_call):
        assert isinstance(options, dict)
        assert options["local_files_only"] is True
        assert options["trust_remote_code"] is False
    assert model.generate.call_args.kwargs["do_sample"] is False
    assert result.reports == {"scan.png": "Findings: no acute abnormality."}
    assert result.display_sections == {"scan.png": {"raw_report": "Findings: no acute abnormality."}}
    assert {
        key: progress[0][3][0][key]
        for key in (
            "filename",
            "original_dimensions",
            "processed_tensor_dimensions",
            "processor_loader",
            "model_loader",
            "adapter",
        )
    } == {
        "filename": "scan.png",
        "original_dimensions": {"width": 3, "height": 2},
        "processed_tensor_dimensions": [1, 3, 896, 896],
        "processor_loader": "auto",
        "model_loader": "image_text_to_text",
        "adapter": "medgemma",
    }

###############################################################################
def test_provider_rejects_unpinned_revision() -> None:
    manifest = _manifest()
    manifest["revision"] = "main"
    image = InferenceImage(filename="scan.png", content_type="image/png", data=_png(), size_bytes=69)

    try:
        HuggingFaceProvider(_settings()).generate(
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
        HuggingFaceProvider(_settings()).generate(
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

###############################################################################
def test_cancellation_after_generation_discards_partial_output(monkeypatch) -> None:
    model = MagicMock()
    model.device = torch.device("cpu")
    processor = MagicMock()
    processor.apply_chat_template.return_value = _processor_inputs()
    stop_requested = False

    def generate(**_kwargs):
        nonlocal stop_requested
        stop_requested = True
        return torch.tensor([[1, 2, 3]])

    model.generate.side_effect = generate
    processor.decode.return_value = "Partial report"
    _patch_runtime(monkeypatch, model, processor)
    progress: list[object] = []

    result = HuggingFaceProvider(_settings()).generate(
        repository_id="google/medgemma-1.5-4b-it",
        manifest=_manifest(),
        profile="deterministic",
        clinical_context="",
        images=[InferenceImage(filename="scan.png", content_type="image/png", data=_png(), size_bytes=69)],
        should_stop=lambda: stop_requested,
        report_progress=lambda *values: progress.append(values),
    )

    assert result.reports == {}
    assert result.display_sections == {}
    assert result.metadata == []
    assert progress == []

###############################################################################
def test_exif_transpose_rgb_conversion_and_processed_dimensions(monkeypatch) -> None:
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    processor = MagicMock()
    processor.apply_chat_template.return_value = _processor_inputs()
    processor.decode.return_value = "Report text"
    _patch_runtime(monkeypatch, model, processor)

    image = Image.new("L", (2, 3), 128)
    exif = image.getexif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())

    progress: list[tuple[object, ...]] = []
    result = HuggingFaceProvider(_settings()).generate(
        repository_id="google/medgemma-1.5-4b-it",
        manifest=_manifest(),
        profile="deterministic",
        clinical_context="",
        images=[InferenceImage(filename="scan.jpg", content_type="image/jpeg", data=buffer.getvalue(), size_bytes=69)],
        should_stop=lambda: False,
        report_progress=lambda *values: progress.append(values),
    )

    messages = processor.apply_chat_template.call_args.args[0]
    processed_image = messages[0]["content"][0]["image"]
    assert processed_image.mode == "RGB"
    assert processed_image.size == (3, 2)
    assert result.metadata[0]["original_dimensions"] == {"width": 3, "height": 2}
    assert result.metadata[0]["processed_tensor_dimensions"] == [1, 3, 8, 9]
    assert progress

###############################################################################
def test_switching_models_and_unload_clear_resident_provider_state(monkeypatch, tmp_path) -> None:
    model_a = MagicMock()
    model_b = MagicMock()
    processor_a = MagicMock()
    processor_b = MagicMock()
    models = [model_a, model_b]
    processors = [processor_a, processor_b]
    monkeypatch.setattr(
        "server.models.inference.providers.adapters.AutoProcessor.from_pretrained",
        lambda _path, **_kwargs: processors.pop(0),
    )
    monkeypatch.setattr(
        "server.models.inference.providers.adapters.AutoModelForImageTextToText.from_pretrained",
        lambda _path, **_kwargs: models.pop(0),
    )
    provider = HuggingFaceProvider(_settings())
    snapshot_a = tmp_path / "snapshot-a"
    snapshot_b = tmp_path / "snapshot-b"
    snapshot_a.mkdir()
    snapshot_b.mkdir()
    manifest_a = {
        **_manifest(),
        "repository_id": "model-a",
        "revision": "a" * 40,
        "local_snapshot_path": str(snapshot_a),
    }
    manifest_b = {
        **_manifest(),
        "repository_id": "model-b",
        "revision": "b" * 40,
        "local_snapshot_path": str(snapshot_b),
    }

    loaded_a = provider._load(manifest_a)
    assert loaded_a[0] is model_a
    loaded_b = provider._load(manifest_b)
    assert loaded_b[0] is model_b
    assert provider._model is model_b
    assert provider._loaded_key == (
        "model-b",
        "b" * 40,
        "medgemma",
        "auto",
        "image_text_to_text",
    )

    provider.unload()

    assert provider._loaded_key is None
    assert provider._model is None
    assert provider._processor is None
    assert provider._adapter is None
