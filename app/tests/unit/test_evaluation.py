import os

import numpy as np
import pandas as pd
import pytest
import torch

os.environ["KERAS_BACKEND"] = "torch"

from server.services import evaluation as evaluation_service
from server.services import validation_runs
from server.services.jobs import JobExecutionError

###############################################################################
class DummyTextGenerator:

    # -------------------------------------------------------------------------
    def __init__(self, model, model_metadata, max_report_size) -> None:
        self.model = model
        self.model_metadata = model_metadata
        self.max_report_size = max_report_size

    # -------------------------------------------------------------------------
    def generate_radiological_reports(
        self,
        image_paths: list[str],
        method: str = "greedy_search",
    ) -> dict[str, str]:
        return dict.fromkeys(image_paths, "one two three four")

###############################################################################
def test_bleu_score_handles_non_string_reports_gracefully(monkeypatch) -> None:
    monkeypatch.setattr(evaluation_service, "TextGenerator", DummyTextGenerator)
    evaluator = evaluation_service.CheckpointEvaluator(
        model=None,
        train_config={},
        model_metadata={},
    )
    validation_data = pd.DataFrame(
        {
            "path": ["image1"],
            "text": [np.nan],
        }
    )

    bleu_score = evaluator.calculate_bleu_score(validation_data, num_samples=1)

    assert bleu_score == pytest.approx(0.0)

###############################################################################
def test_preflight_rejects_checkpoint_sequence_length_mismatch() -> None:
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

    with pytest.raises(evaluation_service.CheckpointInputMismatchError, match="input 1"):
        evaluator.preflight_validation_dataset([batch])

###############################################################################
def test_checkpoint_validation_data_uses_checkpoint_dataset(monkeypatch) -> None:
    requested_names: list[str | None] = []
    validation_data = pd.DataFrame({"path": ["image.png"], "tokens": [[1, 2]]})

    class RepositoryStub:
        def load_training_data(self, *, dataset_name=None):
            requested_names.append(dataset_name)
            return pd.DataFrame(), validation_data, {}

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
def test_checkpoint_validation_data_fails_without_associated_dataset() -> None:
    with pytest.raises(JobExecutionError) as failure:
        validation_runs._load_checkpoint_validation_data(
            ["evaluation_report"],
            {"dataset_name": ""},
        )

    assert failure.value.code == "checkpoint_dataset_unavailable"


