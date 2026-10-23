from __future__ import annotations

import os
import pandas as pd
import pytest
import torch
import hashlib
import importlib
import json
from server.services import evaluation as evaluation_service, validation_runs
from server.services.jobs import JobExecutionError, JobManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from pydantic import ValidationError
from server.domain.validation import CheckpointEvaluationRequest, ValidationRequest
from unittest.mock import MagicMock
from server.configurations import ServerSettings
from server.repositories.checkpoints import CheckpointRepository
from server.services.validation_runs import ValidationService
from tests.conftest import run_async_in_thread







os.environ["KERAS_BACKEND"] = "torch"





###############################################################################
def test_preflight_rejects_checkpoint_sequence_length_mismatch() -> None:

    ###############################################################################
    class ModelStub:
        built = True
        input_shape = [(None, 224, 224, 3), (None, 200)]

    evaluator = evaluation_service.CheckpointEvaluator(
        model=ModelStub(),
        train_config={},
        model_metadata={},
    )
    batch = (
        (torch.zeros((1, 224, 224, 3)), torch.zeros((1, 50), dtype=torch.long)),
        torch.zeros((1, 50), dtype=torch.long),
    )

    with pytest.raises(
        evaluation_service.CheckpointInputMismatchError, match="input 1"
    ):
        evaluator.preflight_validation_dataset([batch])

###############################################################################
def test_checkpoint_validation_uses_its_associated_dataset(monkeypatch) -> None:
    requested_names: list[str | None] = []
    validation_data = pd.DataFrame({"path": ["image.png"], "tokens": [[1, 2]]})

    ###############################################################################
    class RepositoryStub:

        # -------------------------------------------------------------------------
        def load_training_data(self, *, dataset_name=None):
            requested_names.append(dataset_name)
            return pd.DataFrame(), validation_data, {}

        # -------------------------------------------------------------------------
        def validate_img_paths(self, data):
            return data

    monkeypatch.setattr(validation_runs, "DatasetRepository", RepositoryStub)

    result = validation_runs._load_checkpoint_validation_data(
        ["evaluation_report"],
        {"dataset_name": "qa_pre_release_small"},
    )

    assert result is validation_data
    assert requested_names == ["qa_pre_release_small"]

###############################################################################
def test_checkpoint_validation_fails_without_associated_dataset() -> None:
    with pytest.raises(JobExecutionError) as failure:
        validation_runs._load_checkpoint_validation_data(
            ["evaluation_report"],
            {"dataset_name": ""},
        )

    assert failure.value.code == "checkpoint_dataset_unavailable"

"""Unit coverage for independent public-model validation cases."""










validation = importlib.import_module("scripts.validate_inference_model")

###############################################################################
def _write_case_manifest(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "case.png"
    image_path.write_bytes(b"image-bytes")
    manifest_path = tmp_path / "cases.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "case-one",
                        "image": image_path.name,
                        "sha256": hashlib.sha256(b"image-bytes").hexdigest(),
                        "clinical_context": "cough",
                        "profile": "detailed",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, image_path

###############################################################################
def test_load_case_manifest_resolves_relative_images_and_profiles(
    tmp_path: Path,
) -> None:
    manifest_path, image_path = _write_case_manifest(tmp_path)

    cases = validation.load_case_manifest(manifest_path)

    assert len(cases) == 1
    assert cases[0].case_id == "case-one"
    assert cases[0].image_path == image_path.resolve()
    assert cases[0].profile == "detailed"
    assert cases[0].clinical_context == "cough"

###############################################################################
def test_load_case_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    manifest_path, image_path = _write_case_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["cases"].append({**payload["cases"][0], "image": image_path.name})
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate case id"):
        validation.load_case_manifest(manifest_path)

###############################################################################
def test_fixture_metadata_rejects_changed_bytes(tmp_path: Path) -> None:
    image_path = tmp_path / "case.png"
    image_path.write_bytes(b"actual")

    with pytest.raises(ValueError, match="does not match"):
        validation._fixture_metadata(  # noqa: SLF001
            image_path,
            b"actual",
            provenance="public fixture",
            deidentification="no identifiers",
            expected_sha256=hashlib.sha256(b"expected").hexdigest(),
        )

###############################################################################
def test_validation_cache_override_isolated_from_canonical_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers.utils.hub as transformers_hub  # pyright: ignore[reportMissingImports]
    from server.models.inference.providers import huggingface as huggingface_module

    original_values = {
        "transformers": transformers_hub.HF_MODULES_CACHE,
        "huggingface": huggingface_module.HF_MODULES_CACHE,
    }
    monkeypatch.setattr(
        transformers_hub, "HF_MODULES_CACHE", original_values["transformers"]
    )
    monkeypatch.setattr(
        huggingface_module, "HF_MODULES_CACHE", original_values["huggingface"]
    )
    modules_cache = tmp_path / "cache"
    monkeypatch.setenv("XREPORT_VALIDATION_CACHE_ROOT", str(modules_cache))

    configured = validation._configure_validation_cache()  # noqa: SLF001

    assert configured == modules_cache / "huggingface" / "modules"
    assert Path(validation.os.environ["HF_MODULES_CACHE"]) == configured

###############################################################################
def test_validate_cached_cases_runs_each_case_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    cases = [
        validation.ValidationCase(
            case_id="first",
            image_path=first,
            profile="deterministic",
            clinical_context="cough",
            expected_sha256=hashlib.sha256(b"first").hexdigest(),
        ),
        validation.ValidationCase(
            case_id="second",
            image_path=second,
            profile="concise",
            clinical_context="dyspnea",
            expected_sha256=hashlib.sha256(b"second").hexdigest(),
        ),
    ]
    selected = SimpleNamespace(
        model_ref="huggingface:test/model",
        provider="huggingface",
        adapter="test",
        status="ready",
        status_message=None,
        model_revision="a" * 40,
        output_sections=["raw_report"],
        model_dump=lambda mode: {"revision": "a" * 40},
    )
    entry = SimpleNamespace(
        model_ref=selected.model_ref,
        repository_id="test/model",
    )
    requests: list[str] = []

    ###############################################################################
    class FakeProvider:

        # -------------------------------------------------------------------------
        def __init__(self, _settings: object) -> None:
            self._load = lambda _manifest: (object(), object(), object())
            self._generate_study = lambda **_kwargs: None

        # -------------------------------------------------------------------------
        def unload(self) -> None:
            return None

        # -------------------------------------------------------------------------
        @staticmethod
        def _runtime_metadata(_model: object) -> dict[str, object]:
            return {"cuda_used": False, "cuda_available": False}

    def fake_run_inference_job(**kwargs: Any) -> dict[str, object]:
        request_id = str(kwargs["request_id"])
        requests.append(request_id)
        image = kwargs["inference_image_store"].get(request_id)[0]
        report = f"Report for {image.filename}"
        metadata = [
            {
                "filename": image.filename,
                "generation_profile": kwargs["generation_profile"],
                "clinical_context": kwargs["clinical_context"],
            }
        ]
        provenance = {
            "model_revision": selected.model_revision,
            "generation_profile": kwargs["generation_profile"],
            "clinical_context": kwargs["clinical_context"],
            "runtime": {"cuda_used": False, "cuda_available": False},
        }
        sections = {"raw_report": report}
        kwargs["repository"].save_generated_reports(
            [{"image": image.filename, "report": report}],
            generation_config={
                "display_sections": {image.filename: sections},
                "provenance": provenance,
            },
        )
        return {
            "reports": {image.filename: report},
            "reports_ordered": [report],
            "report_filenames": [image.filename],
            "count": 1,
            "display_sections": {image.filename: sections},
            "inference_metadata": metadata,
            "provenance": provenance,
        }

    monkeypatch.setattr(
        validation,
        "get_server_settings",
        lambda: SimpleNamespace(inference=SimpleNamespace()),
    )
    monkeypatch.setattr(
        validation,
        "InferenceModelCatalog",
        lambda _settings: SimpleNamespace(
            list_models=lambda: SimpleNamespace(models=[selected])
        ),
    )
    monkeypatch.setattr(validation, "embedded_inference_models", lambda: [entry])
    monkeypatch.setattr(validation, "HuggingFaceProvider", FakeProvider)
    monkeypatch.setattr(
        validation,
        "InferenceRuntimeCoordinator",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(validation, "ModelInstallationManager", lambda: object())
    monkeypatch.setattr(validation, "validation_contract_hash", lambda _entry: "hash")
    monkeypatch.setattr(validation, "run_inference_job", fake_run_inference_job)

    result = validation.validate_cached_cases(
        model_ref=selected.model_ref,
        cases=cases,
        fixture_provenance="public fixture",
        fixture_deidentification="no identifiers",
        repeat_case_id=None,
        write_receipt=True,
        receipt_path=tmp_path / "aggregate-receipt.json",
    )

    assert result["status"] == "passed"
    assert result["checks"]["completed_cases"] == 2
    assert result["checks"]["reload_reuse_requested"] is False
    assert list(result["reports"]) == ["first.png", "second.png"]
    assert requests == ["validation_case_first", "validation_case_second"]
    receipt = json.loads(
        (tmp_path / "aggregate-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["case_count"] == 2
    assert receipt["checks"]["technical_passed"] is True

###############################################################################
def test_case_manifest_cli_dispatches_to_independent_case_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    case = validation.ValidationCase(
        case_id="case-1",
        image_path=tmp_path / "case.png",
        profile="deterministic",
        clinical_context="",
        expected_sha256="a" * 64,
    )

    monkeypatch.setattr(
        validation,
        "_arguments",
        lambda: SimpleNamespace(
            model_ref="huggingface:test/model",
            case_manifest=tmp_path / "cases.json",
            receipt_path=None,
            fixture_provenance="public fixture",
            fixture_deidentification="no identifiers",
        ),
    )
    monkeypatch.setattr(validation, "load_case_manifest", lambda _path: [case])

    def fake_validate_cached_cases(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "passed"}

    monkeypatch.setattr(validation, "validate_cached_cases", fake_validate_cached_cases)

    assert validation.main() == 0
    assert captured["cases"] == [case]
    assert captured["repeat_case_id"] == "case-1"






###############################################################################
def test_dataset_validation_rejects_unknown_metric() -> None:
    with pytest.raises(ValidationError, match="Unsupported validation metrics"):
        ValidationRequest(
            dataset_name="dataset-1",
            metrics=["unknown_metric"],
            sample_size=1.0,
        )

###############################################################################
def test_checkpoint_evaluation_rejects_unknown_metric() -> None:
    with pytest.raises(
        ValidationError,
        match="Unsupported checkpoint evaluation metrics",
    ):
        CheckpointEvaluationRequest(
            checkpoint="checkpoint-1",
            metrics=["unknown_metric"],
            num_samples=10,
        )
















###############################################################################
class FakeValidationDatasetRepository:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        source: pd.DataFrame,
        validated: pd.DataFrame | None = None,
    ) -> None:
        self.source = source
        self.validated = source if validated is None else validated

    # -------------------------------------------------------------------------
    def load_source_dataset(self, **_kwargs: object) -> pd.DataFrame:
        return self.source

    # -------------------------------------------------------------------------
    def validate_img_paths(self, _dataset: pd.DataFrame) -> pd.DataFrame:
        return self.validated

###############################################################################
def _job_manager(job_type: str) -> Any:
    manager = MagicMock(spec=JobManager)
    manager.is_job_running.return_value = False
    manager.start_job.return_value = "job-1"
    manager.get_job_status.return_value = {
        "job_id": "job-1",
        "job_type": job_type,
        "status": "running",
        "progress": 0.0,
        "result": None,
        "error": None,
    }
    return manager

###############################################################################
def _settings(seed: int = 123) -> ServerSettings:
    return cast(
        ServerSettings,
        SimpleNamespace(
            global_settings=SimpleNamespace(seed=seed),
            jobs=SimpleNamespace(polling_interval=2.0),
        ),
    )

###############################################################################
def test_validation_job_fails_when_dataset_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MagicMock(spec=JobManager)
    manager.should_stop.return_value = False
    repository = FakeValidationDatasetRepository(pd.DataFrame())
    monkeypatch.setattr(validation_runs, "get_job_manager", lambda: manager)
    monkeypatch.setattr(validation_runs, "DatasetRepository", lambda: repository)

    with pytest.raises(JobExecutionError) as raised:
        validation_runs.run_validation_job(
            {
                "dataset_name": "missing-dataset",
                "sample_size": 1.0,
                "metrics": ["text_statistics"],
                "seed": 42,
            },
            "job-1",
        )

    assert raised.value.code == "dataset_unavailable"
    assert raised.value.phase == "input_validation"

###############################################################################
def test_validation_job_fails_when_no_image_paths_remain(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MagicMock(spec=JobManager)
    manager.should_stop.return_value = False
    repository = FakeValidationDatasetRepository(
        pd.DataFrame([{"path": "missing.png", "text": "report"}]),
        validated=pd.DataFrame(),
    )
    monkeypatch.setattr(validation_runs, "get_job_manager", lambda: manager)
    monkeypatch.setattr(validation_runs, "DatasetRepository", lambda: repository)

    with pytest.raises(JobExecutionError) as raised:
        validation_runs.run_validation_job(
            {
                "dataset_name": "broken-dataset",
                "sample_size": 1.0,
                "metrics": ["image_statistics"],
                "seed": 42,
            },
            "job-1",
        )

    assert raised.value.code == "dataset_integrity_failed"
    assert raised.value.phase == "input_validation"

###############################################################################
def test_checkpoint_evaluation_persistence_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MagicMock()
    repository.save_checkpoint_evaluation_report.side_effect = RuntimeError(
        "database unavailable"
    )
    monkeypatch.setattr(validation_runs, "ValidationRepository", lambda: repository)

    with pytest.raises(JobExecutionError) as raised:
        validation_runs._save_checkpoint_evaluation_report(
            "checkpoint-1",
            ["evaluation_report"],
            {"evaluation_report": {"data_fraction": 1.0}},
            {"loss": 0.2, "accuracy": 0.8},
        )

    assert raised.value.code == "persistence_failed"
    assert raised.value.phase == "persistence"

###############################################################################
def test_validation_service_preserves_explicit_zero_seed() -> None:
    manager = _job_manager("validation")
    service = ValidationService(manager, _settings())

    run_async_in_thread(
        service.run_validation(
            ValidationRequest(
                dataset_name="dataset-1",
                metrics=["text_statistics"],
                sample_size=1.0,
                seed=0,
            )
        )
    )

    request_data = manager.start_job.call_args.kwargs["kwargs"]["request_data"]
    assert request_data["seed"] == 0

###############################################################################
def test_validation_service_uses_global_seed_when_omitted() -> None:
    manager = _job_manager("validation")
    service = ValidationService(manager, _settings(seed=321))

    run_async_in_thread(
        service.run_validation(
            ValidationRequest(
                dataset_name="dataset-1",
                metrics=["text_statistics"],
                sample_size=1.0,
            )
        )
    )

    request_data = manager.start_job.call_args.kwargs["kwargs"]["request_data"]
    assert request_data["seed"] == 321

###############################################################################
def test_checkpoint_evaluation_uses_global_seed_when_omitted() -> None:
    manager = _job_manager("checkpoint_evaluation")
    checkpoint_repository = MagicMock(spec=CheckpointRepository)
    checkpoint_repository.get_checkpoint.return_value = SimpleNamespace(
        artifact_complete=True
    )
    service = ValidationService(
        manager,
        _settings(seed=654),
        checkpoint_repository=cast(CheckpointRepository, checkpoint_repository),
    )

    run_async_in_thread(
        service.evaluate_checkpoint(
            CheckpointEvaluationRequest(
                checkpoint="checkpoint-1",
                metrics=["evaluation_report"],
                num_samples=10,
            )
        )
    )

    request_data = manager.start_job.call_args.kwargs["kwargs"]["request_data"]
    assert request_data["seed"] == 654

###############################################################################
def test_validation_service_reads_seed_and_poll_interval_for_each_new_run() -> None:
    manager = _job_manager("validation")
    current = _settings(seed=111)
    service = ValidationService(
        manager,
        current,
        settings_provider=lambda: current,
    )

    first_response = run_async_in_thread(
        service.run_validation(
            ValidationRequest(
                dataset_name="dataset-1",
                metrics=["text_statistics"],
                sample_size=1.0,
            )
        )
    )
    current.global_settings.seed = 222
    current.jobs.polling_interval = 4.5
    second_response = run_async_in_thread(
        service.run_validation(
            ValidationRequest(
                dataset_name="dataset-2",
                metrics=["text_statistics"],
                sample_size=1.0,
            )
        )
    )

    requests = [call.kwargs["kwargs"]["request_data"] for call in manager.start_job.call_args_list]
    assert [request["seed"] for request in requests] == [111, 222]
    assert first_response.poll_interval == 2.0
    assert second_response.poll_interval == 4.5
