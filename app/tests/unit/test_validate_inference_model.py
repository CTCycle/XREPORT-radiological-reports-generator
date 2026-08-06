from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import validate_inference_model

###############################################################################
def test_fixture_metadata_requires_matching_hash_and_records_no_image_data() -> None:
    data = b"public-deidentified-fixture"
    digest = hashlib.sha256(data).hexdigest()

    metadata = validate_inference_model._fixture_metadata(
        Path("scan.png"),
        data,
        provenance="Dataset release 1, accession CXR-001",
        deidentification="Dataset documentation confirms de-identification",
        expected_sha256=digest.upper(),
    )

    assert metadata == {
        "filename": "scan.png",
        "provenance": "Dataset release 1, accession CXR-001",
        "de_identification": "Dataset documentation confirms de-identification",
        "sha256": digest,
    }
    assert "public-deidentified-fixture" not in metadata.values()
    with pytest.raises(SystemExit, match="does not match"):
        validate_inference_model._fixture_metadata(
            Path("scan.png"),
            data,
            provenance="Dataset release 1",
            deidentification="De-identified public release",
            expected_sha256="0" * 64,
        )

###############################################################################
def test_non_huggingface_model_returns_deferred_without_manifest_lookup(monkeypatch, capsys) -> None:
    selected = SimpleNamespace(
        model_ref="xreport:checkpoint_epoch_48",
        provider="xreport",
        status="ready",
        model_revision=None,
        status_message=None,
    )
    catalog = SimpleNamespace(
        list_models=lambda: SimpleNamespace(models=[selected]),
    )
    monkeypatch.setattr(
        validate_inference_model,
        "get_server_settings",
        lambda: SimpleNamespace(inference=object()),
    )
    monkeypatch.setattr(
        validate_inference_model,
        "InferenceModelCatalog",
        lambda _settings: catalog,
    )
    monkeypatch.setattr(
        validate_inference_model,
        "_write_run_log",
        lambda _model_ref, _payload: validate_inference_model.ROOT_DIR / "deferred.json",
    )
    monkeypatch.setattr(
        validate_inference_model,
        "_arguments",
        lambda: argparse.Namespace(
            model_ref=selected.model_ref,
            image=Path("does-not-exist.png"),
            profile="deterministic",
            clinical_context="",
        ),
    )

    assert validate_inference_model.main() == 2
    output = capsys.readouterr().out
    assert '"status": "deferred"' in output
    assert "Hugging Face catalogue entries only" in output
