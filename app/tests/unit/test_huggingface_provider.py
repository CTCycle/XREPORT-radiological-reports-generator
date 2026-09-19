from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image
import pytest
import torch
from transformers import StoppingCriteriaList

from server.configurations import InferenceSettings
from server.domain.inference import InferenceImage
from server.models.inference.providers.adapters import (
    MedGemmaAdapter,
    StudyGeneration,
)
from server.models.inference.providers import huggingface as huggingface_module
from server.models.inference.providers.huggingface import HuggingFaceProvider


REVISION = "a" * 40

###############################################################################
def _settings() -> InferenceSettings:
    return InferenceSettings(
        hf_local_only=True,
        device="cpu",
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
    return Inputs(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros((1, 3, 8, 9)),
        }
    )

###############################################################################
def test_generate_uses_manifest_loaders_revision_and_records_dimensions(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        huggingface_module,
        "is_within_allowed_roots",
        lambda path: path.resolve().is_relative_to(tmp_path.resolve()),
    )
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    calls: dict[str, object] = {}
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    processor = MagicMock()
    processor.apply_chat_template.return_value = Inputs(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros((1, 3, 896, 896)),
        }
    )
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
        images=[
            InferenceImage(
                filename="scan.png",
                content_type="image/png",
                data=_png(),
                size_bytes=69,
            )
        ],
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
    assert result.display_sections == {
        "scan.png": {"raw_report": "Findings: no acute abnormality."}
    }
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
    image = InferenceImage(
        filename="scan.png", content_type="image/png", data=_png(), size_bytes=69
    )

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
    image = InferenceImage(
        filename="scan.png", content_type="image/png", data=_png(), size_bytes=69
    )

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
        images=[
            InferenceImage(
                filename="scan.png",
                content_type="image/png",
                data=_png(),
                size_bytes=69,
            )
        ],
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
        images=[
            InferenceImage(
                filename="scan.jpg",
                content_type="image/jpeg",
                data=buffer.getvalue(),
                size_bytes=69,
            )
        ],
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
def test_switching_models_and_unload_clear_resident_provider_state(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        huggingface_module,
        "is_within_allowed_roots",
        lambda path: path.resolve().is_relative_to(tmp_path.resolve()),
    )
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

###############################################################################
def test_generation_reads_timeout_provider_once_for_start_deadline(monkeypatch) -> None:
    timeout_reads = 0

    def read_timeout() -> int:
        nonlocal timeout_reads
        timeout_reads += 1
        return 5

    provider = HuggingFaceProvider(_settings(), timeout_provider=read_timeout)
    adapter = MagicMock()
    adapter.supports_study = False
    deadlines: list[float] = []
    monkeypatch.setattr(
        HuggingFaceProvider,
        "validate_manifest",
        classmethod(lambda _cls, _repository_id, payload: payload),
    )
    monkeypatch.setattr(
        HuggingFaceProvider,
        "_validate_images",
        staticmethod(lambda _repository_id, _manifest, _images: None),
    )
    monkeypatch.setattr(provider, "_load", lambda _manifest: (object(), object(), adapter))
    monkeypatch.setattr(
        provider,
        "_generate_image",
        lambda **kwargs: (
            deadlines.append(kwargs["deadline"]),
            ("report", {}, {"raw_report": "report"}),
        )[1],
    )
    monkeypatch.setattr(provider, "_check_deadline", lambda _repository_id, _deadline: None)
    monkeypatch.setattr(huggingface_module.time, "monotonic", lambda: 100.0)

    result = provider.generate(
        repository_id="model",
        manifest={**_manifest(), "repository_id": "model", "output_sections": ["raw_report"]},
        profile="deterministic",
        clinical_context="",
        images=[InferenceImage("image.png", "image/png", _png(), len(_png()))],
        should_stop=lambda: False,
        report_progress=lambda *_values: None,
    )

    assert result.reports == {"image.png": "report"}
    assert timeout_reads == 1
    assert deadlines == [105.0]


def test_study_generation_runs_in_inference_mode_and_receives_stopping_criteria(
    monkeypatch,
) -> None:
    provider = HuggingFaceProvider(_settings())
    observed: dict[str, object] = {}

    class StudyAdapter:
        supports_study = True

        def generate_study(self, **kwargs: object) -> StudyGeneration:
            observed["grad_enabled"] = torch.is_grad_enabled()
            observed["stopping_criteria"] = kwargs["stopping_criteria"]
            return StudyGeneration(
                report="study report",
                display_sections={"raw_report": "study report"},
                metadata=[{"filename": "image.png"}],
            )

    monkeypatch.setattr(
        HuggingFaceProvider,
        "validate_manifest",
        classmethod(lambda _cls, _repository_id, payload: payload),
    )
    monkeypatch.setattr(
        HuggingFaceProvider,
        "_validate_images",
        staticmethod(lambda _repository_id, _manifest, _images: None),
    )
    model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
    monkeypatch.setattr(
        provider,
        "_load",
        lambda _manifest: (model, object(), StudyAdapter()),
    )

    with torch.enable_grad():
        result = provider.generate(
            repository_id="model",
            manifest={
                **_manifest(),
                "repository_id": "model",
                "output_sections": ["raw_report"],
            },
            profile="deterministic",
            clinical_context="",
            images=[InferenceImage("image.png", "image/png", _png(), len(_png()))],
            should_stop=lambda: False,
            report_progress=lambda *_values: None,
        )

    assert observed["grad_enabled"] is False
    assert isinstance(observed["stopping_criteria"], StoppingCriteriaList)
    assert result.provenance["runtime"] == {
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "resolved_devices": ["cpu"],
        "model_dtype": "torch.float32",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_used": False,
    }


def test_device_policy_and_dtype_selection(monkeypatch) -> None:
    assert HuggingFaceProvider(_settings())._device_map() == "cpu"
    assert HuggingFaceProvider(
        InferenceSettings(hf_local_only=True, device="auto", model_timeout=600)
    )._device_map() == "auto"

    monkeypatch.setattr(huggingface_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        HuggingFaceProvider(
            InferenceSettings(hf_local_only=True, device="cuda", model_timeout=600)
        )._device_map()
    assert HuggingFaceProvider._dtype("auto") is torch.float32

    monkeypatch.setattr(huggingface_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        huggingface_module.torch.cuda, "is_bf16_supported", lambda: True
    )
    assert HuggingFaceProvider._dtype("auto") is torch.bfloat16
    monkeypatch.setattr(
        huggingface_module.torch.cuda, "is_bf16_supported", lambda: False
    )
    assert HuggingFaceProvider._dtype("auto") is torch.float16


def test_move_inputs_preserves_integer_ids_and_casts_floating_inputs() -> None:
    model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float16)
    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
    }

    moved = HuggingFaceProvider._move_inputs(inputs, model)

    assert moved["input_ids"].dtype is torch.long
    assert moved["attention_mask"].dtype is torch.long
    assert moved["pixel_values"].dtype is torch.float16
    assert moved["pixel_values"].device == torch.device("cpu")


def test_move_inputs_casts_nested_generation_inputs() -> None:
    model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16)
    inputs = {
        "time_deltas": [torch.zeros((1, 2), dtype=torch.float32)],
        "input_ids": [torch.tensor([1, 2], dtype=torch.long)],
    }

    moved = HuggingFaceProvider._move_inputs(inputs, model)

    assert moved["time_deltas"][0].dtype is torch.bfloat16
    assert moved["time_deltas"][0].device == torch.device("cpu")
    assert moved["input_ids"][0].dtype is torch.long


def test_move_inputs_uses_accelerate_input_device_map(monkeypatch) -> None:
    model = SimpleNamespace(
        hf_device_map={"": "cpu", "model.layers.0": "cpu"},
        dtype=torch.bfloat16,
    )
    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
    }

    moved = HuggingFaceProvider._move_inputs(inputs, model)

    assert HuggingFaceProvider._input_device(model) == torch.device("cpu")
    assert moved["input_ids"].dtype is torch.long
    assert moved["pixel_values"].dtype is torch.bfloat16
    assert moved["pixel_values"].device == torch.device("cpu")

    monkeypatch.setattr(huggingface_module.torch.cuda, "is_available", lambda: True)
    distributed = SimpleNamespace(
        hf_device_map={"model.embed_tokens": "cuda:0", "lm_head": "cpu"},
        dtype=torch.bfloat16,
    )
    runtime = HuggingFaceProvider(_settings())._runtime_metadata(distributed)
    assert runtime["resolved_devices"] == ["cuda:0", "cpu"]
    assert runtime["resolved_device"] == "cuda:0"
    assert runtime["model_dtype"] == "torch.bfloat16"
    assert runtime["cuda_available"] is True
    assert runtime["cuda_used"] is True


def test_move_inputs_supports_mocked_cuda_without_casting_token_ids(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_to(self, *args: object, **kwargs: object):
        calls.append((args, kwargs))
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    model = SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.float16)
    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
    }

    moved = HuggingFaceProvider._move_inputs(inputs, model)

    assert moved is not inputs
    assert len(calls) == 2
    assert calls[0][1]["dtype"] is torch.long
    assert calls[1][1]["dtype"] is torch.float16
    assert calls[0][1]["device"] == torch.device("cuda:0")


def test_provider_reuses_same_resident_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        huggingface_module,
        "is_within_allowed_roots",
        lambda path: path.resolve().is_relative_to(tmp_path.resolve()),
    )
    model = MagicMock()
    processor = MagicMock()
    load_calls = 0

    def load_model(_path: str, **_kwargs: object) -> MagicMock:
        nonlocal load_calls
        load_calls += 1
        return model

    monkeypatch.setattr(
        "server.models.inference.providers.adapters.AutoProcessor.from_pretrained",
        lambda _path, **_kwargs: processor,
    )
    monkeypatch.setattr(
        "server.models.inference.providers.adapters.AutoModelForImageTextToText.from_pretrained",
        load_model,
    )
    provider = HuggingFaceProvider(_settings())
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = {
        **_manifest(),
        "repository_id": "model",
        "local_snapshot_path": str(snapshot),
    }

    assert provider._load(manifest)[0] is model
    assert provider._load(manifest)[0] is model
    assert load_calls == 1
