from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from server.repositories.database.engine import Database
from server.repositories.schemas import (
    Base,
    Checkpoint,
    CheckpointEvaluation,
    Dataset,
    DatasetRecord,
    DatasetVersion,
    InferenceReport,
    InferenceRun,
    ProcessingRun,
    ValidationRun,
)
from server.repositories.serialization.dataset import DatasetRepository, sample_dataset
from server.repositories.serialization.inference import InferenceRepository
from server.repositories.serialization.validation import ValidationRepository

###############################################################################
def _serializer() -> tuple[DatasetRepository, Database]:
    database = Database.__new__(Database)
    database.engine = create_engine("sqlite:///:memory:", future=True)
    database.session = sessionmaker(
        bind=database.engine, future=True, expire_on_commit=False
    )
    database.insert_batch_size = 100
    Base.metadata.create_all(database.engine)
    return DatasetRepository(database=database), database

###############################################################################
def test_dataset_import_replaces_stale_rows_and_updates_reports() -> None:
    serializer, database = _serializer()
    serializer.upsert_source_dataset(
        pd.DataFrame(
            [
                {
                    "dataset_name": "Chest",
                    "image_name": "A.PNG",
                    "report_text": "old",
                    "image_path": "old-path",
                },
                {
                    "dataset_name": "Chest",
                    "image_name": "B.PNG",
                    "report_text": "removed-on-reload",
                    "image_path": "removed-path",
                },
            ]
        )
    )
    serializer.upsert_source_dataset(
        pd.DataFrame(
            [
                {
                    "dataset_name": " chest ",
                    "image_name": "a.png",
                    "report_text": "new",
                    "image_path": "new-path",
                }
            ]
        )
    )

    with database.read_session() as session:
        dataset = session.execute(select(Dataset)).scalar_one()
        latest = session.execute(
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id == dataset.dataset_id)
            .order_by(DatasetVersion.version_number.desc())
            .limit(1)
        ).scalar_one()
        rows = (
            session.execute(
                select(DatasetRecord).where(
                    DatasetRecord.dataset_version_id == latest.dataset_version_id
                )
            )
            .scalars()
            .all()
        )

    assert dataset.name_key == "chest"
    assert latest.version_number == 2
    assert len(rows) == 1
    assert rows[0].image_name_key == "a.png"
    assert rows[0].report_text == "new"
    assert rows[0].image_path == "new-path"

###############################################################################
def test_fractional_sampling_retains_one_row_for_small_nonempty_dataset() -> None:
    dataset = pd.DataFrame([{"record_id": 1}, {"record_id": 2}])

    sampled = sample_dataset(dataset, sample_size=0.2, seed=42)

    assert len(sampled) == 1
    assert sampled.iloc[0]["record_id"] in {1, 2}

###############################################################################
def test_dataset_import_rolls_back_when_a_record_is_invalid() -> None:
    serializer, database = _serializer()
    serializer.upsert_source_dataset(
        pd.DataFrame(
            [
                {
                    "dataset_name": "Chest",
                    "image_name": "A.PNG",
                    "report_text": "stable",
                    "image_path": "path",
                }
            ]
        )
    )

    invalid_payload = pd.DataFrame(
        [
            {
                "dataset_name": "Chest",
                "image_name": "A.PNG",
                "report_text": "duplicate",
                "image_path": "path",
                "row_order": 1,
            },
            {
                "dataset_name": "Chest",
                "image_name": "A.PNG",
                "report_text": "duplicate-again",
                "image_path": "path",
                "row_order": 2,
            },
        ]
    )

    try:
        serializer.upsert_source_dataset(invalid_payload)
    except Exception:
        pass

    with database.read_session() as session:
        rows = session.execute(select(DatasetRecord)).scalars().all()
    assert len(rows) == 1
    assert rows[0].report_text == "stable"

###############################################################################
def test_identical_dataset_content_reuses_the_existing_version() -> None:
    serializer, database = _serializer()
    payload = pd.DataFrame(
        [
            {
                "dataset_name": "Chest",
                "image_name": "A.PNG",
                "report_text": "stable",
                "image_path": "path",
            }
        ]
    )
    serializer.upsert_source_dataset(payload)
    serializer.upsert_source_dataset(payload.copy())

    with database.read_session() as session:
        versions = session.execute(select(DatasetVersion)).scalars().all()
    assert len(versions) == 1
    assert versions[0].version_number == 1

###############################################################################
def test_processing_run_and_samples_roll_back_together() -> None:
    serializer, database = _serializer()
    serializer.upsert_source_dataset(
        pd.DataFrame(
            [
                {
                    "dataset_name": "Chest",
                    "image_name": "A.PNG",
                    "report_text": "stable",
                    "image_path": "path",
                }
            ]
        )
    )
    with database.read_session() as session:
        record_id = session.execute(select(DatasetRecord.record_id)).scalar_one()

    training_data = pd.DataFrame(
        [
            {
                "record_id": record_id,
                "image": "A.PNG",
                "text": "stable",
                "tokens": [1, 2],
                "split": "train",
                "path": "path",
            },
            {
                "record_id": record_id,
                "image": "A.PNG",
                "text": "stable",
                "tokens": [3, 4],
                "split": "train",
                "path": "path",
            },
        ]
    )
    try:
        serializer.save_training_data(
            {"dataset_name": "processed", "source_dataset": "Chest"},
            training_data,
            hashcode="config-hash",
        )
    except Exception:
        pass

    with database.read_session() as session:
        assert session.execute(select(ProcessingRun)).scalars().all() == []

###############################################################################
def test_validation_report_children_commit_atomically() -> None:
    dataset_repository, database = _serializer()
    dataset_repository.upsert_source_dataset(
        pd.DataFrame(
            [
                {
                    "dataset_name": "Chest",
                    "image_name": "A.PNG",
                    "report_text": "stable",
                    "image_path": "path",
                }
            ]
        )
    )
    ValidationRepository(database=database).save_validation_report(
        {
            "dataset_name": "Chest",
            "sample_size": 1.0,
            "metrics": ["text"],
            "text_statistics": {"count": 1, "total_words": 1},
            "pixel_distribution": {"bins": [0], "counts": [1]},
        }
    )
    with database.read_session() as session:
        assert len(session.execute(select(ValidationRun)).scalars().all()) == 1

###############################################################################
def test_validation_aggregates_are_stored_on_the_run() -> None:
    _, database = _serializer()
    ValidationRepository(database=database).save_validation_report(
        {
            "dataset_name": "Chest",
            "metrics": [],
            "pixel_distribution": {"bins": [999], "counts": [1]},
        }
    )
    with database.read_session() as session:
        run = session.execute(select(ValidationRun)).scalar_one()
        assert run.pixel_bins_json == [999]

###############################################################################
def test_checkpoint_evaluation_is_owned_by_validation_repository() -> None:
    _, database = _serializer()
    with database.transaction() as session:
        session.add(
            Checkpoint(
                name="checkpoint-1",
                name_key="checkpoint-1",
                path="registered-checkpoint",
            )
        )
    repository = ValidationRepository(database=database)
    repository.save_checkpoint_evaluation_report(
        {
            "checkpoint": "checkpoint-1",
            "metrics": ["bleu_score"],
            "metric_configs": {"bleu_score": {"data_fraction": 0.5}},
            "results": {"bleu_score": 0.75},
        }
    )

    with database.read_session() as session:
        evaluations = session.execute(select(CheckpointEvaluation)).scalars().all()

    assert len(evaluations) == 1
    assert repository.get_checkpoint_evaluation_report("checkpoint-1") == {
        "checkpoint": "checkpoint-1",
        "date": evaluations[0].executed_at.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": ["bleu_score"],
        "metric_configs": {"bleu_score": {"data_fraction": 0.5}},
        "results": {"bleu_score": 0.75},
    }

###############################################################################
def test_inference_reports_preserve_input_order_and_are_idempotent() -> None:
    _, database = _serializer()
    serializer = InferenceRepository(database=database)
    serializer.save_generated_reports(
        [
            {"image": "B.PNG", "report": "second"},
            {"image": "A.PNG", "report": "first"},
        ],
        provider="huggingface",
        model_ref="huggingface:google/medgemma-1.5-4b-it",
        model_revision=None,
        generation_profile="deterministic",
        generation_config={"temperature": 0},
        clinical_context="Cough",
        request_id="request-1",
        status="succeeded",
        execution_time_seconds=2.5,
    )
    serializer.save_generated_reports(
        [
            {"image": "A.PNG", "report": "replayed"},
        ],
        provider="huggingface",
        model_ref="huggingface:google/medgemma-1.5-4b-it",
        model_revision=None,
        generation_profile="concise",
        generation_config={"temperature": 0},
        clinical_context="Updated",
        request_id="request-1",
        status="succeeded",
        execution_time_seconds=1.25,
    )
    with database.read_session() as session:
        runs = session.execute(select(InferenceRun)).scalars().all()
        reports = (
            session.execute(
                select(InferenceReport).order_by(InferenceReport.image_index)
            )
            .scalars()
            .all()
        )
    assert len(runs) == 1
    assert runs[0].checkpoint_id is None
    assert runs[0].provider == "huggingface"
    assert runs[0].model_ref == "huggingface:google/medgemma-1.5-4b-it"
    assert runs[0].generation_profile == "concise"
    assert runs[0].clinical_context == "Updated"
    assert runs[0].execution_time_seconds == 1.25
    assert len(reports) == 1
    assert reports[0].image_index == 0
    assert reports[0].generated_report == "replayed"
    assert serializer.list_inference_history(
        model_ref="huggingface:google/medgemma-1.5-4b-it"
    ) == {
        "items": [
            {
                "request_id": "request-1",
                "provider": "huggingface",
                "model_ref": "huggingface:google/medgemma-1.5-4b-it",
                "model_revision": None,
                "generation_profile": "concise",
                "clinical_context": "Updated",
                "status": "succeeded",
                "execution_time_seconds": 1.25,
                "date": runs[0].executed_at.strftime("%Y-%m-%d %H:%M:%S"),
                "reports": [
                    {
                        "image_index": 0,
                        "input_image_name": "A.PNG",
                        "preview": "replayed",
                        "edited": False,
                        "edited_at": None,
                    }
                ],
                "image_names": ["A.PNG"],
                "report_count": 1,
                "provenance_available": False,
            }
        ],
        "total": 1,
        "limit": 50,
        "offset": 0,
    }


###############################################################################
def test_inference_history_crud_preserves_original_text_and_isolated_deletion() -> None:
    _, database = _serializer()
    repository = InferenceRepository(database=database)
    repository.save_generated_reports(
        [
            {"image": "older.png", "report": "Findings\nClear lungs"},
            {"image": "second.png", "report": "Findings\nNo acute disease"},
        ],
        provider="huggingface",
        model_ref="huggingface:example/model",
        model_revision="a" * 40,
        generation_profile="deterministic",
        generation_config={
            "display_sections": {
                "older.png": {"findings": "Clear lungs"},
                "second.png": {"findings": "No acute disease"},
            },
            "provenance": {"model_revision": "a" * 40},
        },
        clinical_context="Follow-up",
        request_id="older-request",
        status="succeeded",
        execution_time_seconds=1.0,
        executed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    repository.save_generated_reports(
        [{"image": "newer.png", "report": "Impression\nStable"}],
        provider="huggingface",
        model_ref="huggingface:example/model",
        model_revision="b" * 40,
        generation_profile="concise",
        generation_config={},
        clinical_context="",
        request_id="newer-request",
        status="failed",
        execution_time_seconds=None,
        executed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    newest = repository.list_inference_history(limit=1)
    assert newest["total"] == 2
    assert newest["items"][0]["request_id"] == "newer-request"
    oldest_failed = repository.list_inference_history(status="failed", sort="oldest")
    assert [item["request_id"] for item in oldest_failed["items"]] == ["newer-request"]

    detail = repository.get_inference_history("older-request")
    assert detail is not None
    assert [report["image_index"] for report in detail["reports"]] == [0, 1]
    assert detail["reports"][0]["sections"] == {"findings": "Clear lungs"}
    assert detail["reports"][0]["generated_report"] == "Findings\nClear lungs"

    with pytest.raises(ValueError, match="unique"):
        repository.update_inference_reports(
            "older-request",
            [
                {"image_index": 0, "edited_report": "one"},
                {"image_index": 0, "edited_report": "two"},
            ],
        )
    with pytest.raises(ValueError, match="Unknown"):
        repository.update_inference_reports(
            "older-request", [{"image_index": 9, "edited_report": "missing"}]
        )

    updated = repository.update_inference_reports(
        "older-request", [{"image_index": 0, "edited_report": "Edited draft"}]
    )
    assert updated is not None
    assert updated["reports"][0]["generated_report"] == "Findings\nClear lungs"
    assert updated["reports"][0]["edited_report"] == "Edited draft"
    assert updated["reports"][0]["effective_report"] == "Edited draft"

    cleared = repository.update_inference_reports(
        "older-request",
        [{"image_index": 0, "edited_report": "Findings\nClear lungs"}],
    )
    assert cleared is not None
    assert cleared["reports"][0]["edited_report"] is None
    assert cleared["reports"][0]["edited_at"] is None
    assert repository.delete_inference_history("older-request") is True
    assert repository.get_inference_history("older-request") is None
    assert repository.get_inference_history("newer-request") is not None
