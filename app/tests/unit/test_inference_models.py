from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock
from PIL import Image
import pytest
import threading
import time
import torch
import json
import zipfile
import numpy as np
from transformers import StoppingCriteriaList
from xreport_shared.configurations import InferenceSettings
from xreport_inference.domain.inference import InferenceImage
from xreport_inference.models.inference.providers.adapters import MedGemmaAdapter, StudyGeneration
from xreport_inference.models.inference.providers import huggingface as huggingface_module, xreport
from xreport_inference.models.inference.providers.huggingface import HuggingFaceProvider
from xreport_inference.configurations.inference_models import embedded_inference_models
from xreport_inference.services.inference_runtime import InferenceRuntimeCoordinator
from xreport_inference.services.inference_catalog import validation_contract_hash, InferenceModelCatalog
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from xreport_shared.common.path import ROOT_DIR
from xreport_inference.services.model_installation import ModelInstallationManager, InstallationError
from xreport_shared.common import path as path_module
from xreport_shared.common.runtime_layout import RuntimeLayout
from xreport_inference.services import model_installation as installation_module
from xreport_inference.services.model_storage import ModelStorageLifecycle
from xreport_inference.models.inference.providers.xreport import XReportCheckpointProvider




















HUGGINGFACE_REVISION = "a" * 40

###############################################################################
def _huggingface_settings() -> InferenceSettings:
    return InferenceSettings(
        hf_local_only=True,
        device="cpu",
        model_timeout=600,
        model_idle_timeout=300,
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
def _huggingface_manifest() -> dict[str, object]:
    return {
        "revision": HUGGINGFACE_REVISION,
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
        lambda _self, _huggingface_manifest: (model, processor, MedGemmaAdapter()),
    )
    monkeypatch.setattr(
        "xreport_inference.models.inference.providers.adapters.AutoProcessor.from_pretrained",
        lambda _path, **_kwargs: processor,
    )
    monkeypatch.setattr(
        "xreport_inference.models.inference.providers.adapters.AutoModelForImageTextToText.from_pretrained",
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
def _stored_image() -> InferenceImage:
    return InferenceImage(
        filename="scan.png",
        content_type="image/png",
        data=_png(),
        size_bytes=69,
    )

###############################################################################
def test_validate_display_sections_accepts_non_empty_findings_only_contract() -> None:
    HuggingFaceProvider._validate_display_sections(
        "model",
        _stored_image(),
        {"findings": "No acute cardiopulmonary abnormality."},
        ["findings"],
    )

###############################################################################
def test_validate_display_sections_rejects_empty_findings_only_contract() -> None:
    with pytest.raises(RuntimeError, match="incomplete report sections"):
        HuggingFaceProvider._validate_display_sections(
            "model",
            _stored_image(),
            {"findings": ""},
            ["findings"],
        )

###############################################################################
def test_validate_display_sections_keeps_two_section_contract_strict() -> None:
    with pytest.raises(RuntimeError, match="incomplete report sections"):
        HuggingFaceProvider._validate_display_sections(
            "model",
            _stored_image(),
            {"findings": "Finding text"},
            ["findings", "impression"],
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
        "xreport_inference.models.inference.providers.adapters.AutoProcessor.from_pretrained",
        load_processor,
    )
    monkeypatch.setattr(
        "xreport_inference.models.inference.providers.adapters.AutoModelForImageTextToText.from_pretrained",
        load_model,
    )
    progress: list[tuple[object, ...]] = []

    result = HuggingFaceProvider(_huggingface_settings()).generate(
        repository_id="google/medgemma-1.5-4b-it",
        manifest={**_huggingface_manifest(), "local_snapshot_path": str(snapshot_path.resolve())},
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
    manifest = _huggingface_manifest()
    manifest["revision"] = "main"
    image = InferenceImage(
        filename="scan.png", content_type="image/png", data=_png(), size_bytes=69
    )

    try:
        HuggingFaceProvider(_huggingface_settings()).generate(
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
        HuggingFaceProvider(_huggingface_settings()).generate(
            repository_id="google/medgemma-1.5-4b-it",
            manifest=_huggingface_manifest(),
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

    result = HuggingFaceProvider(_huggingface_settings()).generate(
        repository_id="google/medgemma-1.5-4b-it",
        manifest=_huggingface_manifest(),
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
    result = HuggingFaceProvider(_huggingface_settings()).generate(
        repository_id="google/medgemma-1.5-4b-it",
        manifest=_huggingface_manifest(),
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
        "xreport_inference.models.inference.providers.adapters.AutoProcessor.from_pretrained",
        lambda _path, **_kwargs: processors.pop(0),
    )
    monkeypatch.setattr(
        "xreport_inference.models.inference.providers.adapters.AutoModelForImageTextToText.from_pretrained",
        lambda _path, **_kwargs: models.pop(0),
    )
    provider = HuggingFaceProvider(_huggingface_settings())
    snapshot_a = tmp_path / "snapshot-a"
    snapshot_b = tmp_path / "snapshot-b"
    snapshot_a.mkdir()
    snapshot_b.mkdir()
    manifest_a = {
        **_huggingface_manifest(),
        "repository_id": "model-a",
        "revision": "a" * 40,
        "local_snapshot_path": str(snapshot_a),
    }
    manifest_b = {
        **_huggingface_manifest(),
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
def test_failed_model_load_leaves_no_resident_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        huggingface_module,
        "is_within_allowed_roots",
        lambda path: path.resolve().is_relative_to(tmp_path.resolve()),
    )
    monkeypatch.setattr(
        "xreport_inference.models.inference.providers.adapters.AutoProcessor.from_pretrained",
        lambda _path, **_kwargs: MagicMock(),
    )

    def fail_load(_path, **_kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        "xreport_inference.models.inference.providers.adapters.AutoModelForImageTextToText.from_pretrained",
        fail_load,
    )
    released = MagicMock()
    monkeypatch.setattr(HuggingFaceProvider, "_release_memory", staticmethod(released))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    provider = HuggingFaceProvider(_huggingface_settings())

    with pytest.raises(RuntimeError, match="out of memory"):
        provider._load(
            {
                **_huggingface_manifest(),
                "repository_id": "model-a",
                "local_snapshot_path": str(snapshot),
            }
        )

    released.assert_called_once()
    assert provider.loaded_repository is None
    assert provider._model is None

###############################################################################
def _resident_coordinator(**kwargs: Any) -> tuple[InferenceRuntimeCoordinator, MagicMock]:
    provider = MagicMock()
    provider.loaded_repository = "model-a"
    provider.unload.side_effect = lambda: setattr(provider, "loaded_repository", None)
    coordinator = InferenceRuntimeCoordinator(
        huggingface_provider=provider,
        installation_manager=MagicMock(),
        storage_lifecycle=MagicMock(),
        checkpoint_repository=MagicMock(),
        **kwargs,
    )
    return coordinator, provider

###############################################################################
def test_release_never_interrupts_a_run_holding_the_residency_lock() -> None:
    coordinator, provider = _resident_coordinator()
    run_started = threading.Event()
    finish_run = threading.Event()

    def active_run() -> None:
        with coordinator.lock:
            run_started.set()
            finish_run.wait(5)

    run = threading.Thread(target=active_run)
    run.start()
    run_started.wait(5)
    try:
        assert coordinator.release(blocking=False) is False
        provider.unload.assert_not_called()
        assert coordinator.resident_model == "huggingface:model-a"
    finally:
        finish_run.set()
        run.join(5)

    assert coordinator.release(blocking=False) is True
    assert coordinator.resident_model is None

###############################################################################
def test_idle_watcher_releases_only_after_timeout_and_when_lock_is_free() -> None:
    coordinator, provider = _resident_coordinator(idle_timeout_provider=lambda: 1)
    coordinator.idle_check_interval = 0.01
    coordinator.last_used = time.monotonic() - 10
    with coordinator.lock:
        coordinator._ensure_idle_watcher()
        watcher = coordinator._idle_watcher
        time.sleep(0.1)
        provider.unload.assert_not_called()

    assert watcher is not None
    watcher.join(5)
    provider.unload.assert_called_once()
    assert coordinator.resident_model is None
    assert coordinator._idle_watcher is None

###############################################################################
def test_idle_watcher_keeps_model_when_timeout_is_disabled() -> None:
    coordinator, provider = _resident_coordinator(idle_timeout_provider=lambda: 0)
    coordinator.idle_check_interval = 0.01
    coordinator.last_used = time.monotonic() - 10
    coordinator._ensure_idle_watcher()
    watcher = coordinator._idle_watcher
    time.sleep(0.1)
    provider.unload.assert_not_called()

    assert coordinator.release(blocking=True) is True
    assert watcher is not None
    watcher.join(5)
    assert coordinator._idle_watcher is None

###############################################################################
def test_xreport_checkpoint_stays_resident_until_released(monkeypatch, tmp_path) -> None:
    from xreport_shared.repositories.serialization import model as serializer_module
    from xreport_inference.services import inference_runtime as runtime_module

    framework_release = MagicMock()
    monkeypatch.setattr(runtime_module, "release_framework_memory", framework_release)
    loaded = MagicMock()
    load_checkpoint = MagicMock(return_value=(loaded, {}, {"max_report_size": 10}, {}, ""))
    monkeypatch.setattr(
        serializer_module.ModelSerializer,
        "load_checkpoint",
        lambda self, path: load_checkpoint(path),
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "saved_model.keras").write_bytes(b"weights")
    coordinator, provider = _resident_coordinator()

    first = coordinator._load_xreport("xreport:checkpoint", str(checkpoint))
    second = coordinator._load_xreport("xreport:checkpoint", str(checkpoint))

    assert first[0] is second[0] is loaded
    load_checkpoint.assert_called_once()
    provider.unload.assert_called_once()
    assert coordinator.resident_model == "xreport:checkpoint"

    assert coordinator.release() is True
    assert coordinator.resident_model is None
    assert coordinator._xreport_model is None
    framework_release.assert_called_once_with(keras_session=True)

###############################################################################
def test_generation_reads_timeout_provider_once_for_start_deadline(monkeypatch) -> None:
    timeout_reads = 0

    def read_timeout() -> int:
        nonlocal timeout_reads
        timeout_reads += 1
        return 5

    provider = HuggingFaceProvider(_huggingface_settings(), timeout_provider=read_timeout)
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
        staticmethod(lambda _repository_id, _huggingface_manifest, _images: None),
    )
    monkeypatch.setattr(provider, "_load", lambda _huggingface_manifest: (object(), object(), adapter))
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
        manifest={**_huggingface_manifest(), "repository_id": "model", "output_sections": ["raw_report"]},
        profile="deterministic",
        clinical_context="",
        images=[InferenceImage("image.png", "image/png", _png(), len(_png()))],
        should_stop=lambda: False,
        report_progress=lambda *_values: None,
    )

    assert result.reports == {"image.png": "report"}
    assert timeout_reads == 1
    assert deadlines == [105.0]

###############################################################################
def test_study_generation_runs_in_inference_mode_and_receives_stopping_criteria(
    monkeypatch,
) -> None:
    provider = HuggingFaceProvider(_huggingface_settings())
    observed: dict[str, object] = {}

    ###############################################################################
    class StudyAdapter:
        supports_study = True

        # -------------------------------------------------------------------------
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
        staticmethod(lambda _repository_id, _huggingface_manifest, _images: None),
    )
    model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
    monkeypatch.setattr(
        provider,
        "_load",
        lambda _huggingface_manifest: (model, object(), StudyAdapter()),
    )

    with torch.enable_grad():
        result = provider.generate(
            repository_id="model",
            manifest={
                **_huggingface_manifest(),
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

###############################################################################
def test_device_policy_and_dtype_selection(monkeypatch) -> None:
    assert HuggingFaceProvider(_huggingface_settings())._device_map() == "cpu"
    assert HuggingFaceProvider(
        InferenceSettings(hf_local_only=True, device="auto", model_timeout=600, model_idle_timeout=300)
    )._device_map() == "auto"

    monkeypatch.setattr(huggingface_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        HuggingFaceProvider(
            InferenceSettings(hf_local_only=True, device="cuda", model_timeout=600, model_idle_timeout=300)
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

###############################################################################
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

###############################################################################
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

###############################################################################
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
    runtime = HuggingFaceProvider(_huggingface_settings())._runtime_metadata(distributed)
    assert runtime["resolved_devices"] == ["cuda:0", "cpu"]
    assert runtime["resolved_device"] == "cuda:0"
    assert runtime["model_dtype"] == "torch.bfloat16"
    assert runtime["cuda_available"] is True
    assert runtime["cuda_used"] is True

###############################################################################
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

###############################################################################
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
        "xreport_inference.models.inference.providers.adapters.AutoProcessor.from_pretrained",
        lambda _path, **_kwargs: processor,
    )
    monkeypatch.setattr(
        "xreport_inference.models.inference.providers.adapters.AutoModelForImageTextToText.from_pretrained",
        load_model,
    )
    provider = HuggingFaceProvider(_huggingface_settings())
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = {
        **_huggingface_manifest(),
        "repository_id": "model",
        "local_snapshot_path": str(snapshot),
    }

    assert provider._load(manifest)[0] is model
    assert provider._load(manifest)[0] is model
    assert load_calls == 1







###############################################################################
def test_embedded_catalog_contains_unique_sha_pinned_public_models() -> None:
    models = embedded_inference_models()
    assert models
    assert len({entry.model_ref for entry in models}) == len(models)
    assert all(len(entry.revision) == 40 for entry in models)
    assert {entry.adapter for entry in models} == {
        "cxrmate_multi",
        "chexone",
        "cxrmate2",
        "medgemma",
    }
    assert {entry.model_ref: entry.revision for entry in models} == {
        "huggingface:aehrc/cxrmate-multi-tf": "330721b9aa5bba201a3eb88eba4dd9a6607f3e7a",
        "huggingface:StanfordAIMI/CheXOne": "0c350e6852ea08f9d9baf3b7595c1a10d4849927",
        "huggingface:aehrc/cxrmate-2": "aa8e2d16470e20671acf049687b4707c9bf2f2b5",
        "huggingface:google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    }
    assert {
        entry.model_ref: validation_contract_hash(entry)
        for entry in models
    } == {
        "huggingface:aehrc/cxrmate-multi-tf": "4feea8f90fba6c26881d35e6178fcc878eb56a8ed53df717d74f856d6df92ee7",
        "huggingface:StanfordAIMI/CheXOne": "06314a6921f88ddecbaa25cd9eef44e0cbab78701b06319de11feffbd930dc4e",
        "huggingface:aehrc/cxrmate-2": "cc03947d5f74291f602c9c05e341d58d43fab9883f18637278fe35ee24f0a1c1",
        "huggingface:google/medgemma-1.5-4b-it": "a22ad3bcad11b2438bee9fc4bc53d1cb127d81b42931b0e1192e6170418c0163",
    }

###############################################################################
def test_embedded_chexone_manifest_declares_findings_only() -> None:
    chexone = next(
        entry
        for entry in embedded_inference_models()
        if entry.model_ref == "huggingface:StanfordAIMI/CheXOne"
    )

    assert chexone.output_sections == ["findings"]
    assert chexone.capabilities.findings is True
    assert chexone.capabilities.impression is False
    assert chexone.capabilities.grounding is True

###############################################################################
def test_runtime_rejects_incomplete_model_manifest() -> None:
    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        InferenceRuntimeCoordinator._require_complete_manifest({"revision": "6" * 40})











###############################################################################
def _catalog_settings(*, hf_local_only: bool = True) -> InferenceSettings:
    return InferenceSettings(
        hf_local_only=hf_local_only,
        device="auto",
        model_timeout=600,
        model_idle_timeout=300,
    )

###############################################################################
class _InstallationManager(ModelInstallationManager):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        inspector: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> None:
        self.inspector = inspector

    # -------------------------------------------------------------------------
    def inspect(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        return self.inspector(manifest)

###############################################################################
def _not_installed() -> dict[str, Any]:
    return {
        "metadata": {},
        "state": "not_installed",
        "integrity": "unknown",
        "active_revision": None,
        "active_path": None,
        "candidate": None,
        "candidate_path": None,
        "candidate_revision": None,
    }

###############################################################################
def _catalog(
    checkpoints: list[object],
    inspect_installation: Callable[[Mapping[str, Any]], dict[str, Any]] = (
        lambda _manifest: _not_installed()
    ),
) -> InferenceModelCatalog:
    repository = SimpleNamespace(list_checkpoints=lambda: checkpoints)
    installation_manager = _InstallationManager(inspect_installation)
    return InferenceModelCatalog(
        _catalog_settings(),
        installation_manager=installation_manager,
        checkpoint_repository=repository,
    )

###############################################################################
def _checkpoint(name: str = "checkpoint_epoch_48", *, complete: bool = True) -> object:
    return SimpleNamespace(
        name=name,
        name_key=name.casefold(),
        path=Path("data/models/checkpoints") / name,
        artifact_complete=complete,
    )

###############################################################################
def test_catalog_exposes_available_public_and_custom_model_sources() -> None:
    response = _catalog([_checkpoint()]).list_models()

    public = [model for model in response.models if model.origin == "public"]
    custom = [model for model in response.models if model.origin == "custom"]
    assert public
    assert all(model.provider == "huggingface" for model in public)
    assert [model.model_ref for model in custom] == ["xreport:checkpoint_epoch_48"]
    assert set(response.providers) == {"huggingface", "xreport"}
    assert response.providers["huggingface"].status == "not_installed"
    assert response.providers["xreport"].status == "ready"
    assert all(model.available_actions == ["download"] for model in public)

###############################################################################
def test_catalog_disables_public_models_when_huggingface_runtime_is_disabled() -> None:
    catalog = _catalog([_checkpoint()])
    catalog.settings = _catalog_settings(hf_local_only=False)
    response = catalog.list_models()

    public = [model for model in response.models if model.origin == "public"]
    assert public
    assert all(model.status == "disabled" for model in public)
    assert response.providers["huggingface"].status == "disabled"

###############################################################################
def test_catalog_hides_xreport_provider_without_registered_checkpoints() -> None:
    response = _catalog([]).list_models()

    assert not any(model.provider == "xreport" for model in response.models)
    assert response.providers["xreport"].status == "not_installed"

###############################################################################
def test_catalog_exposes_chexone_findings_only_contract() -> None:
    chexone = next(
        model
        for model in _catalog([]).list_models().models
        if model.model_ref == "huggingface:StanfordAIMI/CheXOne"
    )

    assert chexone.output_sections == ["findings"]
    assert chexone.capabilities.findings is True
    assert chexone.capabilities.impression is False
    assert chexone.provider == "huggingface"
    assert chexone.origin == "public"
    assert chexone.adapter == "chexone"

###############################################################################
def test_catalog_marks_verified_active_installation_ready() -> None:
    active_path = (
        ROOT_DIR
        / "data"
        / "models"
        / "huggingface"
        / "installed"
        / "active"
    )
    response = _catalog(
        [],
        inspect_installation=lambda manifest: {
            "metadata": {},
            "state": "active",
            "integrity": "verified",
            "active_revision": manifest["revision"],
            "active_path": active_path,
            "candidate": None,
            "candidate_path": None,
            "candidate_revision": None,
        },
    )

    model = response.list_models().models[0]

    assert model.status == "ready"
    assert model.installation_state == "active"
    assert model.integrity_status == "verified"
    assert model.local_path == "data/models/huggingface/installed/active"
















INSTALLATION_REVISION = "a" * 40
NEXT_REVISION = "b" * 40

###############################################################################
def _installation_manifest(revision: str = INSTALLATION_REVISION) -> dict[str, object]:
    return {
        "repository_id": "example/report-model",
        "revision": revision,
        "required_files": ["config.json", "tokenizer.json"],
        "weight_file_sets": [["model.safetensors"]],
    }

###############################################################################
def _active_metadata(
    manager: ModelInstallationManager,
    installed: Path,
    **updates: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "repository_id": "example/report-model",
        "state": "active",
        "active_revision": INSTALLATION_REVISION,
        "active_relative_path": manager.relative_path(installed),
        "candidate": None,
        "rollback": None,
    }
    payload.update(updates)
    return payload

###############################################################################
class FakeApi:

    # -------------------------------------------------------------------------
    def __init__(self, revision: str = INSTALLATION_REVISION) -> None:
        self.revision = revision

    # -------------------------------------------------------------------------
    def model_info(self, repository_id: str, **kwargs: object) -> SimpleNamespace:
        del repository_id, kwargs
        return SimpleNamespace(
            sha=self.revision,
            siblings=[
                SimpleNamespace(rfilename="config.json", size=2, lfs=None),
                SimpleNamespace(rfilename="tokenizer.json", size=2, lfs=None),
                SimpleNamespace(
                    rfilename="model.safetensors",
                    size=2,
                    lfs=SimpleNamespace(sha256=""),
                ),
            ],
        )

###############################################################################
class CompleteResponse:
    status_code = 200

    # -------------------------------------------------------------------------
    def __init__(self, body: bytes) -> None:
        self.body = body

    # -------------------------------------------------------------------------
    def raise_for_status(self) -> None:
        return None

    # -------------------------------------------------------------------------
    def iter_content(self, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [self.body]

    # -------------------------------------------------------------------------
    def close(self) -> None:
        return None

###############################################################################
def complete_get(url: str, **_kwargs: object) -> CompleteResponse:
    filename = url.rsplit("/", 1)[-1].split("?", 1)[0]
    return CompleteResponse(b"ok" if filename == "model.safetensors" else b"{}")

###############################################################################
@pytest.fixture
def manager_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "portable"
    data_root = root / "data"
    monkeypatch.setattr(
        path_module,
        "RUNTIME_LAYOUT",
        RuntimeLayout(
            mode="packaged",
            runtime_root=root,
            data_root=data_root,
            resources_root=data_root,
            client_dist_dir=root / "app" / "client",
            release_version="3.1.0",
            variant="cpu",
        ),
    )
    monkeypatch.setattr(installation_module, "ROOT_DIR", root)
    monkeypatch.setattr(installation_module, "DATA_ROOT", data_root)
    monkeypatch.setattr(
        installation_module,
        "HF_STAGING_DIR",
        data_root / "models" / "huggingface" / "staging",
    )
    monkeypatch.setattr(
        installation_module,
        "HF_INSTALLED_DIR",
        data_root / "models" / "huggingface" / "installed",
    )
    monkeypatch.setattr(
        installation_module,
        "HF_ROLLBACK_DIR",
        data_root / "models" / "huggingface" / "rollback",
    )
    monkeypatch.setattr(
        installation_module,
        "HF_METADATA_DIR",
        data_root / "models" / "huggingface" / "metadata",
    )
    for path in (
        data_root,
        installation_module.HF_STAGING_DIR,
        installation_module.HF_INSTALLED_DIR,
        installation_module.HF_ROLLBACK_DIR,
        installation_module.HF_METADATA_DIR,
        root / "runtimes" / "cache" / "huggingface" / "hub",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return root

###############################################################################
def test_stage_downloads_only_approved_files_and_records_verified_candidate(
    manager_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del manager_paths
    calls: list[str] = []

    def fake_get(url: str, **kwargs: object) -> CompleteResponse:
        calls.append(url)
        return complete_get(url, **kwargs)

    monkeypatch.setattr(installation_module.requests, "get", fake_get)
    manager = ModelInstallationManager(api=FakeApi())
    target = manager.stage(
        manifest=_installation_manifest(),
        revision=INSTALLATION_REVISION,
        should_stop=lambda: False,
        report_progress=lambda _payload: None,
    )

    assert target.candidate is True
    assert {url.rsplit("/", 1)[-1].split("?", 1)[0] for url in calls} == {
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    }
    assert all(INSTALLATION_REVISION in url for url in calls)
    metadata = manager.read_metadata("example/report-model")
    assert metadata["state"] == "staged"
    assert metadata["integrity"] == "verified"
    assert metadata["candidate"]["revision"] == INSTALLATION_REVISION

###############################################################################
def test_stage_reuses_interrupted_staging_directory(
    manager_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del manager_paths
    manager = ModelInstallationManager(api=FakeApi())
    partial = (
        installation_module.HF_STAGING_DIR
        / "resume-op"
        / "example__report-model"
        / INSTALLATION_REVISION
    )
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(installation_module.requests, "get", complete_get)

    target = manager.stage(
        manifest=_installation_manifest(),
        revision=INSTALLATION_REVISION,
        should_stop=lambda: False,
        report_progress=lambda _payload: None,
    )

    assert target.path == partial

###############################################################################
def test_http_downloader_resumes_partial_weight_file(
    manager_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del manager_paths
    manager = ModelInstallationManager(api=FakeApi())
    partial = (
        installation_module.HF_STAGING_DIR
        / "resume-http"
        / "example__report-model"
        / INSTALLATION_REVISION
        / "model.safetensors.incomplete"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"o")
    calls: list[dict[str, object]] = []

    ###############################################################################
    class FakeResponse:
        status_code = 206

        # -------------------------------------------------------------------------
        def __init__(self, body: bytes) -> None:
            self.body = body

        # -------------------------------------------------------------------------
        def raise_for_status(self) -> None:
            return None

        # -------------------------------------------------------------------------
        def iter_content(self, chunk_size: int) -> list[bytes]:
            del chunk_size
            return [self.body]

        # -------------------------------------------------------------------------
        def close(self) -> None:
            return None

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        calls.append(kwargs)
        filename = url.rsplit("/", 1)[-1].split("?", 1)[0]
        return FakeResponse(b"k" if filename == "model.safetensors" else b"{}")

    monkeypatch.setattr(installation_module.requests, "get", fake_get)
    target = manager.stage(
        manifest=_installation_manifest(),
        revision=INSTALLATION_REVISION,
        should_stop=lambda: False,
        report_progress=lambda _payload: None,
    )

    assert target.path == partial.parent
    assert (partial.parent / "model.safetensors").read_bytes() == b"ok"
    assert any(call.get("headers") == {"Range": "bytes=1-"} for call in calls)

###############################################################################
def test_check_update_reports_update_only_against_an_active_revision(
    manager_paths: Path,
) -> None:
    del manager_paths
    manager = ModelInstallationManager(api=FakeApi(NEXT_REVISION))

    not_installed = manager.check_update("example/report-model")
    assert not_installed["update_available"] is False
    assert not_installed["latest_revision"] == NEXT_REVISION

    installed = (
        installation_module.HF_INSTALLED_DIR / "example__report-model" / INSTALLATION_REVISION
    )
    installed.mkdir(parents=True)
    manager._write_metadata(
        "example/report-model", _active_metadata(manager, installed)
    )
    outdated = manager.check_update("example/report-model")
    assert outdated["update_available"] is True
    assert outdated["installed_revision"] == INSTALLATION_REVISION

###############################################################################
def test_gated_file_download_reports_access_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = installation_module.requests.Response()
    response.status_code = 401

    def fake_get(url: str, **kwargs: object) -> object:
        del url, kwargs
        return response

    monkeypatch.setattr(installation_module.requests, "get", fake_get)

    with pytest.raises(InstallationError, match="accepted model terms"):
        ModelInstallationManager._open_download_response(
            url="https://huggingface.co/example/report-model/resolve/x/config.json",
            headers={},
            start=0,
            partial=Path("config.json.incomplete"),
            name="config.json",
        )

###############################################################################
def test_activation_preserves_previous_revision_for_rollback(
    manager_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del manager_paths
    manager = ModelInstallationManager(api=FakeApi())
    monkeypatch.setattr(installation_module.requests, "get", complete_get)

    first = manager.stage(
        manifest=_installation_manifest(),
        revision=INSTALLATION_REVISION,
        should_stop=lambda: False,
        report_progress=lambda _payload: None,
    )
    manager.activate(manifest=_installation_manifest(), target=first)
    second = manager.stage(
        manifest=_installation_manifest(NEXT_REVISION),
        revision=NEXT_REVISION,
        should_stop=lambda: False,
        report_progress=lambda _payload: None,
    )
    metadata = manager.activate(manifest=_installation_manifest(NEXT_REVISION), target=second)

    assert metadata["active_revision"] == NEXT_REVISION
    assert metadata["rollback"]["revision"] == INSTALLATION_REVISION
    assert (installation_module.HF_ROLLBACK_DIR / "example__report-model").exists()

###############################################################################
def test_corrupt_active_snapshot_is_marked_and_rejected(manager_paths: Path) -> None:
    del manager_paths
    manager = ModelInstallationManager(api=FakeApi())
    installed = (
        installation_module.HF_INSTALLED_DIR / "example__report-model" / INSTALLATION_REVISION
    )
    installed.mkdir(parents=True)
    for filename in ("config.json", "tokenizer.json", "model.safetensors"):
        (installed / filename).write_text("ok", encoding="utf-8")
    manager._write_metadata(
        "example/report-model",
        _active_metadata(
            manager,
            installed,
            file_manifest={"model.safetensors": {"size": 999, "sha256": "bad"}},
        ),
    )

    with pytest.raises(InstallationError, match="integrity mismatch"):
        manager.active_target(_installation_manifest())

    metadata = manager.read_metadata("example/report-model")
    assert metadata["state"] == "corrupt"
    assert metadata["integrity"] == "failed"

###############################################################################
def test_cancelled_maintenance_preserves_working_active_revision(
    manager_paths: Path,
) -> None:
    del manager_paths
    manager = ModelInstallationManager(api=FakeApi())
    installed = (
        installation_module.HF_INSTALLED_DIR / "example__report-model" / INSTALLATION_REVISION
    )
    installed.mkdir(parents=True)
    for filename in ("config.json", "tokenizer.json", "model.safetensors"):
        (installed / filename).write_text("ok", encoding="utf-8")
    manager._write_metadata(
        "example/report-model",
        _active_metadata(manager, installed, integrity="verified"),
    )
    partial = (
        installation_module.HF_STAGING_DIR
        / "cancelled-op"
        / "example__report-model"
        / INSTALLATION_REVISION
    )
    partial.mkdir(parents=True)
    (partial / "model.safetensors.incomplete").write_bytes(b"partial")

    manager.record_error("example/report-model", "cancelled", interrupted=True)

    metadata = manager.read_metadata("example/report-model")
    assert metadata["state"] == "active"
    assert metadata["active_revision"] == INSTALLATION_REVISION
    assert metadata["interruption"]["resumable"] is True
    assert not partial.exists()

###############################################################################
def test_failed_first_install_keeps_resumable_staging(manager_paths: Path) -> None:
    del manager_paths
    manager = ModelInstallationManager(api=FakeApi())
    partial = (
        installation_module.HF_STAGING_DIR
        / "first-use-op"
        / "example__report-model"
        / INSTALLATION_REVISION
    )
    partial.mkdir(parents=True)
    (partial / "model.safetensors.incomplete").write_bytes(b"partial")

    manager.record_error("example/report-model", "cancelled", interrupted=True)

    assert partial.exists()

###############################################################################
def test_delete_local_removes_only_repository_owned_storage_and_reports_bytes(
    manager_paths: Path,
) -> None:
    manager = ModelInstallationManager(api=FakeApi())
    slug = "example__report-model"
    installed = installation_module.HF_INSTALLED_DIR / slug / INSTALLATION_REVISION
    rollback = installation_module.HF_ROLLBACK_DIR / slug / NEXT_REVISION
    staged = installation_module.HF_STAGING_DIR / "operation" / slug / INSTALLATION_REVISION
    hub_cache = manager_paths / "runtimes" / "cache" / "huggingface" / "hub"
    cache = hub_cache / "models--example--report-model" / "snapshots" / INSTALLATION_REVISION
    unrelated = (
        installation_module.HF_STAGING_DIR / "operation" / "other__model" / INSTALLATION_REVISION
    )
    for path in (installed, rollback, staged, cache, unrelated):
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(
            b"owned" if path != unrelated else b"keep"
        )
    manager._write_metadata(
        "example/report-model",
        _active_metadata(manager, installed),
    )

    deleted = ModelStorageLifecycle(
        manager,
        root_dir=installation_module.ROOT_DIR,
        installed_dir=installation_module.HF_INSTALLED_DIR,
        hub_cache_dir=hub_cache,
        rollback_dir=installation_module.HF_ROLLBACK_DIR,
        staging_dir=installation_module.HF_STAGING_DIR,
    ).delete_local("example/report-model")

    assert deleted["state"] == "not_installed"
    assert deleted["bytes_freed"] > 0
    assert not installed.parent.exists()
    assert not rollback.parent.exists()
    assert not staged.parent.exists()
    assert not cache.parent.parent.exists()
    assert not manager._metadata_path("example/report-model").exists()
    assert unrelated.exists()












###############################################################################
def _write_checkpoint_files(checkpoint_dir: Path, *, valid_model: bool) -> None:
    configuration_dir = checkpoint_dir / "configuration"
    configuration_dir.mkdir(parents=True)
    for name in ("configuration.json", "metadata.json", "session_history.json"):
        (configuration_dir / name).write_text(json.dumps({}), encoding="utf-8")
    model_path = checkpoint_dir / "saved_model.keras"
    if valid_model:
        with zipfile.ZipFile(model_path, "w") as archive:
            archive.writestr("metadata.json", "{}")
    else:
        model_path.write_bytes(b"not-a-keras-archive")

###############################################################################
def test_checkpoint_validation_accepts_complete_and_rejects_incomplete_or_corrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "saved_model.keras").write_bytes(b"not-a-keras-archive")
    corrupt = tmp_path / "corrupt"
    _write_checkpoint_files(corrupt, valid_model=False)
    complete = tmp_path / "complete"
    _write_checkpoint_files(complete, valid_model=True)
    provider = XReportCheckpointProvider()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        provider.validate_checkpoint(incomplete)
    with pytest.raises(ValueError, match="invalid Keras archive"):
        provider.validate_checkpoint(corrupt)
    assert provider.validate_checkpoint(complete) == "complete"

###############################################################################
def test_generate_rejects_empty_checkpoint_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    ###############################################################################
    class FakeModel:

        # -------------------------------------------------------------------------
        def summary(self, *, expand_nested: bool) -> None:
            del expand_nested

    ###############################################################################
    class FakeGenerator:
        generator_image_methods = {"greedy_search": lambda *args, **kwargs: ""}

        # -------------------------------------------------------------------------
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        # -------------------------------------------------------------------------
        def load_tokenizer_and_configuration(self):
            return FakeTokenizer(), {}

    ###############################################################################
    class FakeTokenizer:

        # -------------------------------------------------------------------------
        def get_vocab(self) -> dict[str, int]:
            return {"[CLS]": 0}

    ###############################################################################
    class FakeDataLoader:

        # -------------------------------------------------------------------------
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        # -------------------------------------------------------------------------
        def prepare_inference_image_bytes(self, data: bytes) -> np.ndarray:
            del data
            return np.zeros((224, 224, 3), dtype=np.float32)

    monkeypatch.setattr(xreport, "TextGenerator", FakeGenerator)
    monkeypatch.setattr(xreport, "XRAYDataLoader", FakeDataLoader)

    with pytest.raises(RuntimeError, match="empty or malformed"):
        XReportCheckpointProvider().generate(
            model=FakeModel(),
            model_metadata={},
            generation_mode="greedy_search",
            images=[InferenceImage("scan.png", "image/png", b"bytes", 5)],
            should_stop=lambda: False,
            report_progress=lambda *args: None,
        )
