from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from server.domain.inference import InferenceImage
from server.models.inference.providers import xreport
from server.models.inference.providers.xreport import XReportCheckpointProvider


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
