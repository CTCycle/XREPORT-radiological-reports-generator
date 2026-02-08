import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from XREPORT.server.services import evaluation as evaluation_service


class DummyTextGenerator:
    def __init__(self, model, model_metadata, max_report_size) -> None:
        self.model = model
        self.model_metadata = model_metadata
        self.max_report_size = max_report_size

    def generate_radiological_reports(
        self,
        image_paths: list[str],
        method: str = "greedy_search",
    ) -> dict[str, str]:
        return dict.fromkeys(image_paths, "one two three four")


def test_bleu_score_skips_non_string_reports(monkeypatch) -> None:
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


def test_bleu_score_accepts_valid_string_reports(monkeypatch) -> None:
    monkeypatch.setattr(evaluation_service, "TextGenerator", DummyTextGenerator)
    evaluator = evaluation_service.CheckpointEvaluator(
        model=None,
        train_config={},
        model_metadata={},
    )
    validation_data = pd.DataFrame(
        {
            "path": ["image1"],
            "text": ["one two three four"],
        }
    )

    bleu_score = evaluator.calculate_bleu_score(validation_data, num_samples=1)

    assert 0.0 < bleu_score <= 1.0
