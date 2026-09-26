"""Unit coverage for independent public-model validation cases."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

validation = importlib.import_module("scripts.validate_inference_model")


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


def test_load_case_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    manifest_path, image_path = _write_case_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["cases"].append({**payload["cases"][0], "image": image_path.name})
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate case id"):
        validation.load_case_manifest(manifest_path)


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


def test_validation_cache_override_isolated_from_canonical_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers.utils.hub as transformers_hub  # pyright: ignore[reportMissingImports]
    from server.models.inference.providers import adapters as adapters_module
    from server.models.inference.providers import huggingface as huggingface_module

    original_values = {
        "transformers": transformers_hub.HF_MODULES_CACHE,
        "adapters": adapters_module.HF_MODULES_CACHE,
        "huggingface": huggingface_module.HF_MODULES_CACHE,
    }
    monkeypatch.setattr(
        transformers_hub, "HF_MODULES_CACHE", original_values["transformers"]
    )
    monkeypatch.setattr(
        adapters_module, "HF_MODULES_CACHE", original_values["adapters"]
    )
    monkeypatch.setattr(
        huggingface_module, "HF_MODULES_CACHE", original_values["huggingface"]
    )
    modules_cache = tmp_path / "cache"
    monkeypatch.setenv("XREPORT_VALIDATION_CACHE_ROOT", str(modules_cache))

    configured = validation._configure_validation_cache()  # noqa: SLF001

    assert configured == modules_cache / "huggingface" / "modules"
    assert Path(validation.os.environ["HF_MODULES_CACHE"]) == configured


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

    class FakeProvider:
        def __init__(self, _settings: object) -> None:
            self._load = lambda _manifest: (object(), object(), object())
            self._generate_study = lambda **_kwargs: None

        def unload(self) -> None:
            return None

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


def test_legacy_validation_path_remains_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        validation,
        "_arguments",
        lambda: SimpleNamespace(
            model_ref="huggingface:test/model",
            case_manifest=None,
            image=[tmp_path / "case.png"],
            profile="deterministic",
            clinical_context="",
            fixture_provenance="public fixture",
            fixture_deidentification="no identifiers",
            fixture_sha256=["a" * 64],
        ),
    )

    def fake_validate_cached_model(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "passed"}

    monkeypatch.setattr(validation, "validate_cached_model", fake_validate_cached_model)

    assert validation.main() == 0
    assert captured["image_paths"] == [tmp_path / "case.png"]
    assert captured["fixture_sha256"] == ["a" * 64]
