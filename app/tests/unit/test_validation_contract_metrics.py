from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.domain.validation import CheckpointEvaluationRequest, ValidationRequest


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
