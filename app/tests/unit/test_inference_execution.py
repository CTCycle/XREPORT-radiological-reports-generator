from __future__ import annotations

from PIL import Image
import pytest
import torch
from server.domain.inference import InferenceImage, ProviderGenerationResult
from server.models.inference.providers import adapters as adapters_module
from server.models.inference.providers.adapters import CheXOneAdapter, CXRMate2Adapter, CXRMateMultiAdapter, MedGemmaAdapter, StudyImage, _ensure_cxrmate2_generation_dtype
from typing import Any, cast
from unittest.mock import MagicMock
from server.services.inference import InferenceImageStore, run_inference_job, InferenceService
from server.services.jobs import JobExecutionError, JobManager, JobState
from types import SimpleNamespace
from server.services.inference_runtime import InferenceRuntimeCoordinator
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from server.api.errors import register_service_error_handlers
from server.api.inference import InferenceEndpoint
from server.configurations import ServerSettings
from server.services.errors import ConflictError
from server.services.inference_catalog import InferenceModelCatalog
from server.services.model_installation import ModelInstallationManager
















###############################################################################
def _study_images() -> list[StudyImage]:
    return [
        StudyImage(
            stored=InferenceImage("a.png", "image/png", b"a", 1),
            image=Image.new("RGB", (4, 4), color=(0, 0, 0)),
            original_dimensions=(4, 4),
        ),
        StudyImage(
            stored=InferenceImage("b.png", "image/png", b"b", 1),
            image=Image.new("RGB", (4, 4), color=(255, 255, 255)),
            original_dimensions=(4, 4),
        ),
    ]

###############################################################################
@pytest.mark.parametrize(
    ("profile", "max_length", "num_beams"),
    (("deterministic", 256, 1), ("concise", 160, 1), ("detailed", 256, 4)),
)
def test_cxrmate_multi_uses_all_images_profile_and_stopping_criteria(
    profile: str, max_length: int, num_beams: int
) -> None:

    ###############################################################################
    class ImageProcessorStub:
        size = {"shortest_edge": 4}
        image_mean = (0.5, 0.5, 0.5)
        image_std = (0.5, 0.5, 0.5)

    ###############################################################################
    class TokenizerStub:
        sep_token_id = 10
        bos_token_id = 11
        eos_token_id = 12
        pad_token_id = 13

    ###############################################################################
    class ModelStub:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.generate_call: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def generate(self, **kwargs: object):
            self.generate_call = kwargs
            return type("Generation", (), {"sequences": torch.tensor([[1, 2]])})()

        # -------------------------------------------------------------------------
        def split_and_decode_sections(self, _output, _token_ids, _tokenizer):
            return ["finding"], ["impression"]

    model = ModelStub()
    stopping_criteria = object()
    result = CXRMateMultiAdapter().generate_study(
        model=model,
        processor={
            "image_processor": ImageProcessorStub(),
            "tokenizer": TokenizerStub(),
        },
        images=_study_images(),
        profile=profile,  # type: ignore[arg-type]
        clinical_context="",
        move_inputs=lambda inputs, _model: inputs,
        stopping_criteria=stopping_criteria,
        output_sections=["findings", "impression"],
    )

    pixel_values = model.generate_call["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (1, 2, 3, 4, 4)
    assert not torch.equal(pixel_values[0, 0], pixel_values[0, 1])
    assert model.generate_call["max_length"] == max_length
    assert model.generate_call["num_beams"] == num_beams
    assert model.generate_call["do_sample"] is False
    assert model.generate_call["stopping_criteria"] is stopping_criteria
    assert result.display_sections == {
        "findings": "finding",
        "impression": "impression",
    }

###############################################################################
@pytest.mark.parametrize(
    ("profile", "max_length", "num_beams"),
    (("deterministic", 256, 1), ("concise", 160, 1), ("detailed", 256, 4)),
)
def test_cxrmate2_forwards_study_and_profile_to_published_processor(
    profile: str, max_length: int, num_beams: int
) -> None:

    ###############################################################################
    class ProcessorStub:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.process_call: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def __call__(self, **kwargs: object):
            self.process_call = kwargs
            return {
                "pixel_values": torch.zeros((1, 2, 3, 4, 4)),
                "input_ids": torch.ones((1, 2), dtype=torch.long),
            }

        # -------------------------------------------------------------------------
        def split_and_decode_sections(self, _generated_ids):
            return ["finding"], ["impression"]

    ###############################################################################
    class ModelStub:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.generate_call: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def generate(self, **kwargs: object):
            self.generate_call = kwargs
            return torch.tensor([[1, 2]])

    processor = ProcessorStub()
    model = ModelStub()
    stopping_criteria = object()
    result = CXRMate2Adapter().generate_study(
        model=model,
        processor=processor,
        images=_study_images(),
        profile=profile,  # type: ignore[arg-type]
        clinical_context="portable chest",
        move_inputs=lambda inputs, _model: inputs,
        stopping_criteria=stopping_criteria,
        output_sections=["findings", "impression"],
    )

    assert processor.process_call["images"] == [[item.image for item in _study_images()]]
    assert processor.process_call["indication"] == "portable chest"
    assert model.generate_call["max_length"] == max_length
    assert model.generate_call["num_beams"] == num_beams
    assert model.generate_call["do_sample"] is False
    assert model.generate_call["stopping_criteria"] is stopping_criteria
    assert result.display_sections == {
        "findings": "finding",
        "impression": "impression",
    }

###############################################################################
def test_cxrmate2_casts_generation_time_deltas_to_model_dtype() -> None:

    ###############################################################################
    class ModelStub:
        device = torch.device("cpu")
        dtype = torch.bfloat16

        # -------------------------------------------------------------------------
        def prepare_inputs_for_generation(self, *_args, **_kwargs):
            return {"time_deltas": torch.zeros((1, 2), dtype=torch.float32)}

    model = ModelStub()
    _ensure_cxrmate2_generation_dtype(
        model,
        lambda inputs, _model: {
            key: value.to(dtype=torch.bfloat16)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
            for key, value in inputs.items()
        },
    )

    prepared = model.prepare_inputs_for_generation(torch.ones((1, 1), dtype=torch.long))

    assert prepared["time_deltas"].dtype is torch.bfloat16

###############################################################################
@pytest.mark.parametrize(
    ("profile", "max_new_tokens"),
    (
        ("deterministic", 768),
        ("concise", 384),
        ("detailed", 1024),
    ),
)
def test_chexone_uses_published_multi_image_vision_path_and_strips_prompt(
    monkeypatch, profile: str, max_new_tokens: int
) -> None:
    images = _study_images()
    vision_call: dict[str, object] = {}

    def process_vision_info(messages):
        vision_call["messages"] = messages
        return ([item.image for item in images], None)

    monkeypatch.setattr(adapters_module, "process_vision_info", process_vision_info)

    ###############################################################################
    class ProcessorStub:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.template_messages = None
            self.processor_call: dict[str, object] = {}
            self.decoded_ids = None

        # -------------------------------------------------------------------------
        def apply_chat_template(self, messages, **_kwargs: object) -> str:
            self.template_messages = messages
            return "rendered chat prompt"

        # -------------------------------------------------------------------------
        def __call__(self, **kwargs: object):
            self.processor_call = kwargs

            ###############################################################################
            class VisionInputs(dict[str, torch.Tensor]):
                input_ids = torch.tensor([[1, 2]])

                # -------------------------------------------------------------------------
                def __init__(self) -> None:
                    super().__init__(
                        input_ids=self.input_ids,
                        pixel_values=torch.zeros((1, 2, 3, 4, 4)),
                    )

            return VisionInputs()

        # -------------------------------------------------------------------------
        def batch_decode(self, generated_ids, **_kwargs: object) -> list[str]:
            self.decoded_ids = generated_ids
            return [
                "No focal airspace opacity. No pleural effusion or pneumothorax."
            ]

    ###############################################################################
    class ModelStub:

        # -------------------------------------------------------------------------
        def generate(self, **kwargs: object):
            self.generate_call = kwargs
            return [torch.tensor([1, 2, 3, 4])]

    processor = ProcessorStub()
    model = ModelStub()
    stopping_criteria = object()
    result = CheXOneAdapter().generate_study(
        model=model,
        processor=processor,
        images=images,
        profile=profile,  # type: ignore[arg-type]
        clinical_context="cough",
        move_inputs=lambda inputs, _model: inputs,
        stopping_criteria=stopping_criteria,
        output_sections=["findings"],
    )

    assert vision_call["messages"] == processor.template_messages
    assert processor.processor_call["images"] == [item.image for item in images]
    assert processor.processor_call["text"] == ["rendered chat prompt"]
    prompt = processor.template_messages[0]["content"][-1]["text"]
    assert "Write only the Findings section" in prompt
    assert "do not add a separate Impression section" in prompt
    assert "or reasoning trace" in prompt
    assert "final Findings and Impression" not in prompt
    assert "Clinical context: cough" in prompt
    assert model.generate_call["max_new_tokens"] == max_new_tokens
    assert model.generate_call["do_sample"] is False
    assert model.generate_call["stopping_criteria"] is stopping_criteria
    assert torch.equal(processor.decoded_ids[0], torch.tensor([3, 4]))
    assert result.report == (
        "No focal airspace opacity. No pleural effusion or pneumothorax."
    )
    assert result.display_sections == {
        "findings": "No focal airspace opacity. No pleural effusion or pneumothorax."
    }

###############################################################################
@pytest.mark.parametrize(
    ("profile", "max_new_tokens"),
    (("deterministic", 768), ("concise", 384), ("detailed", 1536)),
)
def test_medgemma_uses_multi_image_context_raw_report_contract_and_profile(
    profile: str, max_new_tokens: int
) -> None:
    images = _study_images()

    ###############################################################################
    class ProcessorStub:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.messages = None

        # -------------------------------------------------------------------------
        def apply_chat_template(self, messages, **_kwargs: object):
            self.messages = messages
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": torch.zeros((1, 2, 3, 4, 4)),
            }

        # -------------------------------------------------------------------------
        def decode(self, generated, **_kwargs: object) -> str:
            assert torch.equal(generated, torch.tensor([3, 4]))
            return "raw report"

    ###############################################################################
    class ModelStub:

        # -------------------------------------------------------------------------
        def generate(self, **kwargs: object):
            self.generate_call = kwargs
            return [torch.tensor([1, 2, 3, 4])]

    processor = ProcessorStub()
    model = ModelStub()
    stopping_criteria = object()
    result = MedGemmaAdapter().generate_study(
        model=model,
        processor=processor,
        images=images,
        profile=profile,  # type: ignore[arg-type]
        clinical_context="cough",
        move_inputs=lambda inputs, _model: inputs,
        stopping_criteria=stopping_criteria,
        output_sections=["raw_report"],
    )

    assert len(processor.messages[0]["content"]) == 3
    assert processor.messages[0]["content"][-1]["text"].find("cough") >= 0
    assert model.generate_call["max_new_tokens"] == max_new_tokens
    assert model.generate_call["do_sample"] is False
    assert model.generate_call["stopping_criteria"] is stopping_criteria
    assert result.report == "raw report"
    assert result.display_sections == {"raw_report": "raw report"}










###############################################################################
def _image() -> InferenceImage:
    return InferenceImage(
        filename="scan.png",
        content_type="image/png",
        data=b"fixture",
        size_bytes=7,
    )

###############################################################################
def _setup_job(
    manager: JobManager,
    image_store: InferenceImageStore,
    *,
    job_id: str,
    request_id: str,
) -> None:
    manager.jobs[job_id] = JobState(
        job_id=job_id,
        job_type="inference",
        status="running",
    )
    image_store.store(request_id, [_image()])
    image_store.link_job(job_id, request_id)

###############################################################################
def _runtime_stub(provider: Any) -> Any:

    ###############################################################################
    class RuntimeStub:

        # -------------------------------------------------------------------------
        def generate(self, **kwargs):
            generation = provider.generate(**kwargs)
            return ProviderGenerationResult(
                reports=generation.reports,
                display_sections=generation.display_sections,
                metadata=generation.metadata,
                provenance=generation.provenance,
            )

    return RuntimeStub()

###############################################################################
def test_successful_generation_result_exposes_persisted_request_id() -> None:
    manager = JobManager()
    image_store = InferenceImageStore()
    job_id = "generation-result-request-id"
    request_id = "persisted-request-id"
    _setup_job(manager, image_store, job_id=job_id, request_id=request_id)
    provider = MagicMock()
    provider.generate.return_value = ProviderGenerationResult(
        reports={"scan.png": "Findings\nClear lungs."},
        display_sections={"scan.png": {"findings": "Clear lungs."}},
        metadata=[],
        provenance={"provider": "huggingface"},
    )
    repository = MagicMock()

    result = run_inference_job(
        model_ref="huggingface:test/model",
        model_revision="a" * 40,
        model_manifest={"revision": "a" * 40},
        generation_profile="deterministic",
        clinical_context="",
        request_id=request_id,
        job_id=job_id,
        job_manager=manager,
        inference_image_store=image_store,
        runtime=_runtime_stub(provider),
        repository=repository,
    )

    assert result["request_id"] == request_id
    repository.save_generated_reports.assert_called_once()
    assert repository.save_generated_reports.call_args.kwargs["request_id"] == request_id

###############################################################################
def test_per_image_mode_runs_each_image_and_keeps_going_after_a_failure() -> None:
    manager = JobManager()
    image_store = InferenceImageStore()
    job_id, request_id = "per-image-job", "per-image-request"
    manager.jobs[job_id] = JobState(job_id=job_id, job_type="inference", status="running")
    names = ["a.png", "b.png", "c.png"]
    image_store.store(
        request_id,
        [InferenceImage(filename=name, content_type="image/png", data=b"x", size_bytes=1) for name in names],
    )
    image_store.link_job(job_id, request_id)
    seen: list[list[str]] = []

    def generate(**kwargs: Any) -> ProviderGenerationResult:
        filenames = [image.filename for image in kwargs["images"]]
        seen.append(filenames)
        if filenames == ["b.png"]:
            raise RuntimeError("decode failed")
        return ProviderGenerationResult(
            reports={filenames[0]: f"Findings\n{filenames[0]}"},
            display_sections={filenames[0]: {"findings": filenames[0]}},
            metadata=[{"filename": filenames[0]}],
            provenance={"provider": "huggingface", "report_scope": "study"},
        )

    provider = MagicMock()
    provider.generate.side_effect = generate
    repository = MagicMock()

    result = run_inference_job(
        model_ref="huggingface:test/model",
        model_revision="a" * 40,
        model_manifest={"revision": "a" * 40},
        generation_profile="deterministic",
        clinical_context="",
        request_id=request_id,
        job_id=job_id,
        batch_mode="per_image",
        job_manager=manager,
        inference_image_store=image_store,
        runtime=_runtime_stub(provider),
        repository=repository,
    )

    assert seen == [["a.png"], ["b.png"], ["c.png"]]
    assert result["report_filenames"] == ["a.png", "c.png"]
    assert result["provenance"]["report_scope"] == "image"
    assert result["provenance"]["image_errors"] == {"b.png": "decode failed"}
    saved = repository.save_generated_reports.call_args.args[0]
    assert [row["image"] for row in saved] == ["a.png", "c.png"]

###############################################################################
def test_cancelled_after_generation_does_not_persist_partial_reports() -> None:
    manager = JobManager()
    image_store = InferenceImageStore()
    job_id = "cancel-after-generation"
    request_id = "request-cancelled"
    _setup_job(manager, image_store, job_id=job_id, request_id=request_id)

    ###############################################################################
    class CancellingProvider:

        # -------------------------------------------------------------------------
        def generate(self, **_kwargs):
            manager.cancel_job(job_id)
            return ProviderGenerationResult(
                reports={"scan.png": "Partial report"},
                display_sections={"scan.png": {"raw_report": "Partial report"}},
                metadata=[],
                provenance={"provider": "huggingface"},
            )

    repository = MagicMock()
    runtime = _runtime_stub(CancellingProvider())

    result = run_inference_job(
        model_ref="huggingface:test/model",
        model_revision="a" * 40,
        model_manifest={"revision": "a" * 40},
        generation_profile="deterministic",
        clinical_context="",
        request_id=request_id,
        job_id=job_id,
        job_manager=manager,
        inference_image_store=image_store,
        runtime=runtime,
        repository=repository,
    )

    assert result["reports"] == {}
    repository.save_generated_reports.assert_not_called()
    assert image_store.get(request_id) is None

###############################################################################
def test_timeout_does_not_persist_reports() -> None:
    manager = JobManager()
    image_store = InferenceImageStore()
    job_id = "timeout-no-persistence"
    request_id = "request-timeout"
    _setup_job(manager, image_store, job_id=job_id, request_id=request_id)
    provider = MagicMock()
    provider.generate.side_effect = TimeoutError("inference deadline exceeded")
    repository = MagicMock()
    runtime = _runtime_stub(provider)

    with pytest.raises(TimeoutError, match="deadline exceeded"):
        run_inference_job(
            model_ref="huggingface:test/model",
            model_revision="a" * 40,
            model_manifest={"revision": "a" * 40},
            generation_profile="deterministic",
            clinical_context="",
            request_id=request_id,
            job_id=job_id,
            job_manager=manager,
            inference_image_store=image_store,
            runtime=runtime,
            repository=repository,
        )

    repository.save_generated_reports.assert_not_called()
    assert image_store.get(request_id) is None

###############################################################################
def test_persistence_failure_fails_inference_job() -> None:
    manager = JobManager()
    image_store = InferenceImageStore()
    job_id = "persistence-failure"
    request_id = "request-persistence-failure"
    _setup_job(manager, image_store, job_id=job_id, request_id=request_id)
    provider = MagicMock()
    provider.generate.return_value = ProviderGenerationResult(
        reports={"scan.png": "Generated report"},
        display_sections={"scan.png": {"raw_report": "Generated report"}},
        metadata=[],
        provenance={"provider": "huggingface"},
    )
    repository = MagicMock()
    repository.save_generated_reports.side_effect = RuntimeError("database unavailable")
    runtime = _runtime_stub(provider)

    with pytest.raises(JobExecutionError, match="could not be persisted") as exc_info:
        run_inference_job(
            model_ref="huggingface:test/model",
            model_revision="a" * 40,
            model_manifest={"revision": "a" * 40},
            generation_profile="deterministic",
            clinical_context="",
            request_id=request_id,
            job_id=job_id,
            job_manager=manager,
            inference_image_store=image_store,
            runtime=runtime,
            repository=repository,
        )

    assert exc_info.value.code == "persistence_failed"
    assert exc_info.value.phase == "persistence"
    assert image_store.get(request_id) is None







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



















###############################################################################
def test_generation_rejects_degraded_public_models_before_starting_a_job() -> None:
    model_ref = "huggingface:future/degraded-model"
    model = SimpleNamespace(
        model_ref=model_ref,
        origin="public",
        validation_status="degraded",
        status="ready",
    )
    catalog = cast(
        InferenceModelCatalog,
        SimpleNamespace(list_models=lambda: SimpleNamespace(models=[model])),
    )
    job_manager = MagicMock(spec=JobManager)
    service = InferenceService(
        job_manager=job_manager,
        inference_image_store=InferenceImageStore(),
        server_settings=cast(ServerSettings, SimpleNamespace()),
        model_catalog=catalog,
        installation_manager=cast(ModelInstallationManager, MagicMock()),
        repository=MagicMock(),
    )

    with pytest.raises(
        ConflictError,
        match="Model failed qualification and cannot be used for inference",
    ):
        service.generate_reports(
            model_ref=model_ref,
            generation_profile="deterministic",
            clinical_context="",
            images=[],
        )

    job_manager.start_job.assert_not_called()

###############################################################################
def test_inference_api_returns_conflict_for_a_degraded_public_model() -> None:
    model_ref = "huggingface:future/degraded-model"
    model = SimpleNamespace(
        model_ref=model_ref,
        origin="public",
        validation_status="degraded",
        status="ready",
    )
    catalog = cast(
        InferenceModelCatalog,
        SimpleNamespace(list_models=lambda: SimpleNamespace(models=[model])),
    )
    job_manager = MagicMock(spec=JobManager)
    service = InferenceService(
        job_manager=job_manager,
        inference_image_store=InferenceImageStore(),
        server_settings=cast(ServerSettings, SimpleNamespace()),
        model_catalog=catalog,
        installation_manager=cast(ModelInstallationManager, MagicMock()),
        repository=MagicMock(),
    )
    application = FastAPI()
    register_service_error_handlers(application)
    router = InferenceEndpoint(router=APIRouter(prefix="/inference"), service=service)
    router.add_routes()
    application.include_router(router.router)

    with TestClient(application) as client:
        response = client.post(
            "/inference/generate",
            data={
                "model_ref": model_ref,
                "generation_profile": "deterministic",
                "clinical_context": "",
            },
            files={"images": ("scan.png", b"image", "image/png")},
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Model failed qualification and cannot be used for inference: "
        f"{model_ref}"
    }
    job_manager.start_job.assert_not_called()

###############################################################################
def test_generation_rejects_gated_public_models_before_starting_a_job() -> None:
    model_ref = "huggingface:google/medgemma-1.5-4b-it"
    model = SimpleNamespace(
        model_ref=model_ref,
        origin="public",
        validation_status="pending",
        gated=True,
        access_policy="gated",
        status="not_installed",
    )
    catalog = cast(
        InferenceModelCatalog,
        SimpleNamespace(list_models=lambda: SimpleNamespace(models=[model])),
    )
    job_manager = MagicMock(spec=JobManager)
    service = InferenceService(
        job_manager=job_manager,
        inference_image_store=InferenceImageStore(),
        server_settings=cast(ServerSettings, SimpleNamespace()),
        model_catalog=catalog,
        installation_manager=cast(ModelInstallationManager, MagicMock()),
        repository=MagicMock(),
    )

    with pytest.raises(
        ConflictError,
        match="Gated model requires authorized access and a local installation",
    ):
        service.generate_reports(
            model_ref=model_ref,
            generation_profile="deterministic",
            clinical_context="",
            images=[],
        )

    job_manager.start_job.assert_not_called()

###############################################################################
def test_generation_allows_a_ready_gated_public_model() -> None:
    model_ref = "huggingface:google/medgemma-1.5-4b-it"
    model = SimpleNamespace(
        model_ref=model_ref,
        origin="public",
        validation_status="pending",
        gated=True,
        access_policy="gated",
        provider="huggingface",
        status="ready",
        capabilities=SimpleNamespace(clinical_context=False),
        max_current_images=1,
        model_revision="91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
        model_dump=lambda *args, **kwargs: {"model_ref": model_ref},
    )
    catalog = cast(
        InferenceModelCatalog,
        SimpleNamespace(list_models=lambda: SimpleNamespace(models=[model])),
    )
    job_manager = MagicMock(spec=JobManager)
    job_manager.start_job.return_value = "job-1"
    job_manager.get_job_status.return_value = {
        "job_type": "inference",
        "status": "pending",
    }
    service = InferenceService(
        job_manager=job_manager,
        inference_image_store=InferenceImageStore(),
        server_settings=cast(
            ServerSettings, SimpleNamespace(jobs=SimpleNamespace(polling_interval=1.0))
        ),
        model_catalog=catalog,
        installation_manager=cast(ModelInstallationManager, MagicMock()),
        runtime=cast(InferenceRuntimeCoordinator, MagicMock()),
        repository=MagicMock(),
    )

    service.generate_reports(
        model_ref=model_ref,
        generation_profile="deterministic",
        clinical_context="",
        images=[],
    )

    job_manager.start_job.assert_called_once()

###############################################################################
@pytest.mark.parametrize("job_type", ["inference", "model_maintenance"])
def test_runtime_release_is_refused_while_inference_or_download_runs(job_type: str) -> None:
    job_manager = JobManager()
    job_manager.jobs["active"] = JobState(job_id="active", job_type=job_type, status="running")
    runtime = MagicMock()
    service = InferenceService(
        job_manager=job_manager,
        inference_image_store=InferenceImageStore(),
        server_settings=cast(ServerSettings, SimpleNamespace()),
        model_catalog=cast(InferenceModelCatalog, MagicMock()),
        installation_manager=cast(ModelInstallationManager, MagicMock()),
        runtime=cast(InferenceRuntimeCoordinator, runtime),
        repository=MagicMock(),
    )

    with pytest.raises(ConflictError):
        service.release_runtime_memory()
    runtime.release.assert_not_called()

    job_manager.jobs["active"].update(status="completed")
    runtime.resident_model = "huggingface:model-a"
    runtime.release.return_value = True
    response = service.release_runtime_memory()
    assert response.released is True
    assert response.model_ref == "huggingface:model-a"
