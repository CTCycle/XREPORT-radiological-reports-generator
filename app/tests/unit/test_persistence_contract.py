from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from server.repositories.database.engine import Database
from server.repositories.database.initializer import validate_current_schema
from server.repositories.schemas import (
    Base,
    Dataset,
    DatasetRecord,
    DatasetVersion,
    InferenceReport,
    InferenceRun,
    ProcessingRun,
    ValidationRun,
)
from server.repositories.serialization.dataset import DatasetRepository
from server.repositories.serialization.inference import InferenceRepository


def _serializer() -> tuple[DatasetRepository, Database]:
    database = Database.__new__(Database)
    database.engine = create_engine("sqlite:///:memory:", future=True)
    database.session = sessionmaker(
        bind=database.engine, future=True, expire_on_commit=False
    )
    database.insert_batch_size = 100
    Base.metadata.create_all(database.engine)
    return DatasetRepository(database=database), database


def test_inference_first_schema_rejects_create_all_legacy_shape() -> None:
    database = Database.__new__(Database)
    database.engine = create_engine("sqlite:///:memory:", future=True)
    with database.engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE inference_runs ("
                "inference_run_id INTEGER PRIMARY KEY, checkpoint_id INTEGER, "
                "generation_mode TEXT, request_id TEXT)"
            )
        )

    with pytest.raises(ValueError, match="create_all does not migrate columns"):
        validate_current_schema(database)


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
        rows = session.execute(
            select(DatasetRecord).where(
                DatasetRecord.dataset_version_id == latest.dataset_version_id
            )
        ).scalars().all()

    assert dataset.name_key == "chest"
    assert latest.version_number == 2
    assert len(rows) == 1
    assert rows[0].image_name_key == "a.png"
    assert rows[0].report_text == "new"
    assert rows[0].image_path == "new-path"


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


def test_validation_report_children_commit_atomically() -> None:
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
    serializer.save_validation_report(
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


def test_validation_aggregates_are_stored_on_the_run() -> None:
    serializer, database = _serializer()
    serializer.save_validation_report(
        {"dataset_name": "Chest", "metrics": [], "pixel_distribution": {"bins": [999], "counts": [1]}}
    )
    with database.read_session() as session:
        run = session.execute(select(ValidationRun)).scalar_one()
        assert run.pixel_bins_json == [999]


def test_inference_reports_preserve_input_order_and_are_idempotent() -> None:
    _, database = _serializer()
    serializer = InferenceRepository(database=database)
    serializer.save_generated_reports(
        [
            {"image": "B.PNG", "report": "second"},
            {"image": "A.PNG", "report": "first"},
        ],
        provider="ollama",
        model_ref="ollama:medgemma:4b",
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
        provider="ollama",
        model_ref="ollama:medgemma:4b",
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
        reports = session.execute(
            select(InferenceReport).order_by(InferenceReport.image_index)
        ).scalars().all()
    assert len(runs) == 1
    assert runs[0].checkpoint_id is None
    assert runs[0].provider == "ollama"
    assert runs[0].model_ref == "ollama:medgemma:4b"
    assert runs[0].generation_profile == "concise"
    assert runs[0].clinical_context == "Updated"
    assert runs[0].execution_time_seconds == 1.25
    assert len(reports) == 1
    assert reports[0].image_index == 0
    assert reports[0].generated_report == "replayed"
    assert serializer.list_inference_history(model_ref="ollama:medgemma:4b") == [
        {
            "request_id": "request-1",
            "provider": "ollama",
            "model_ref": "ollama:medgemma:4b",
            "model_revision": None,
            "generation_profile": "concise",
            "generation_config": {"temperature": 0},
            "clinical_context": "Updated",
            "status": "succeeded",
            "execution_time_seconds": 1.25,
            "date": runs[0].executed_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]
