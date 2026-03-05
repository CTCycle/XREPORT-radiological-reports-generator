from __future__ import annotations

import os
import types
from typing import Any

import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

from XREPORT.server.learning.device import DeviceDataLoader
from XREPORT.server.learning.training.dataloader import XRAYDataLoader
from XREPORT.server.learning.training.trainer import ModelTrainer
from XREPORT.server.services import processing


###############################################################################
class FakeLoader:
    def __init__(self, length: int) -> None:
        self.length = length

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return self.length

    # -------------------------------------------------------------------------
    def __iter__(self):
        for _ in range(self.length):
            yield ((0, 0), 0)


###############################################################################
class FakeSession:
    def __init__(self) -> None:
        self.history = {"loss": [1.0], "val_loss": [1.2]}
        self.epoch = [0]


###############################################################################
class FakeModel:
    def __init__(self) -> None:
        self.fit_x: Any = None
        self.fit_validation_data: Any = None
        self.fit_steps_per_epoch: int = 0
        self.fit_validation_steps: int = 0

    # -------------------------------------------------------------------------
    def fit(self, x: Any, **kwargs: Any) -> FakeSession:
        self.fit_x = x
        self.fit_validation_data = kwargs.get("validation_data")
        self.fit_steps_per_epoch = int(kwargs.get("steps_per_epoch", 0))
        self.fit_validation_steps = int(kwargs.get("validation_steps", 0))
        return FakeSession()


###############################################################################
class TokenizerCallGuard:
    def __init__(self) -> None:
        self.call_count = 0

    # -------------------------------------------------------------------------
    def fail_if_called(self, *args: Any, **kwargs: Any) -> None:
        self.call_count += 1
        raise AssertionError("Tokenizer loading must not happen in XRAYDataLoader init")


# -----------------------------------------------------------------------------
def test_model_trainer_uses_finite_iterables_for_fit() -> None:
    trainer = ModelTrainer({"training_seed": 42, "epochs": 1, "use_device_GPU": False})
    model = FakeModel()
    train_data = FakeLoader(length=3)
    validation_data = FakeLoader(length=2)

    trained_model, history = trainer.train_model(
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        checkpoint_path=".",
    )

    assert trained_model is model
    assert history["epochs"] == 1
    assert isinstance(model.fit_x, DeviceDataLoader)
    assert isinstance(model.fit_validation_data, DeviceDataLoader)
    assert not isinstance(model.fit_x, types.GeneratorType)
    assert not isinstance(model.fit_validation_data, types.GeneratorType)
    assert model.fit_steps_per_epoch == 3
    assert model.fit_validation_steps == 2


# -----------------------------------------------------------------------------
def test_xray_dataloader_init_does_not_load_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = TokenizerCallGuard()
    monkeypatch.setattr(
        processing.AutoTokenizer,
        "from_pretrained",
        guard.fail_if_called,
    )

    for _ in range(20):
        XRAYDataLoader(
            {
                "batch_size": 2,
                "inference_batch_size": 2,
                "dataloader_workers": 0,
            }
        )

    assert guard.call_count == 0
