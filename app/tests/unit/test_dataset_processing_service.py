from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from server.repositories.serialization.dataset import DatasetRepository
from server.services import dataset_processing
from server.services.dataset_processing import DatasetProcessingService
from server.services.jobs import JobManager

###############################################################################
def test_dataset_processing_uses_injected_runtime_dependencies(monkeypatch) -> None:
    repository = MagicMock(spec=DatasetRepository)
    repository.load_source_dataset.return_value = pd.DataFrame(
        {
            "text": ["first report", "second report"],
            "record_id": [1, 2],
        }
    )
    repository.generate_hashcode.return_value = "hash"
    job_manager = MagicMock(spec=JobManager)
    job_manager.should_stop.return_value = False

    class SanitizerStub:
        def __init__(self, _configuration) -> None:
            pass

        def sanitize_text(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.copy()

    class TokenizerStub:
        tokenizer_id = "stub"
        vocabulary_size = 17

        def __init__(self, _configuration) -> None:
            pass

        def tokenize_text_corpus(self, data: pd.DataFrame) -> pd.DataFrame:
            result = data.copy()
            result["tokens"] = [[1], [2]]
            return result

    class SplitStub:
        def __init__(self, _configuration, data: pd.DataFrame) -> None:
            self.data = data

        def split_train_and_validation(self) -> pd.DataFrame:
            result = self.data.copy()
            result["split"] = ["train", "validation"]
            return result

    monkeypatch.setattr(dataset_processing, "TextSanitizer", SanitizerStub)
    monkeypatch.setattr(dataset_processing, "TokenizerHandler", TokenizerStub)
    monkeypatch.setattr(dataset_processing, "TrainValidationSplit", SplitStub)

    configuration = {
        "dataset_name": "source",
        "custom_name": "processed",
        "sample_size": 1.0,
        "seed": 42,
        "validation_size": 0.5,
        "tokenizer": "stub",
        "max_report_size": 200,
    }
    service = DatasetProcessingService(repository, job_manager)

    result = service.run(configuration, "job-1")

    assert result == {
        "total_samples": 2,
        "train_samples": 1,
        "validation_samples": 1,
        "vocabulary_size": 17,
    }
    assert configuration["dataset_name"] == "processed"
    assert configuration["source_dataset"] == "source"
    repository.load_source_dataset.assert_called_once_with(
        sample_size=1.0,
        seed=42,
        dataset_name="source",
    )
    repository.save_training_data.assert_called_once()
    assert job_manager.update_progress.call_args_list == [
        (("job-1", 30.0),),
        (("job-1", 60.0),),
        (("job-1", 80.0),),
        (("job-1", 100.0),),
    ]
