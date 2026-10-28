from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
import pytest
import pandas as pd
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from xreport_shared.api.errors import register_service_error_handlers
from xreport_studio.api.preparation import PreparationEndpoint
from xreport_shared.configurations import ServerSettings
from xreport_shared.repositories.database import Database as DatasetDatabase
from xreport_studio.repositories.preparation import PreparationRepository
from xreport_inference.repositories.schemas.inference import InferenceBase, InferenceReport, InferenceRun
from xreport_studio.repositories.schemas.studio import StudioBase, Dataset, DatasetRecord, ProcessingRun, TrainingSample, Checkpoint, CheckpointEvaluation, DatasetVersion, ValidationRun
from xreport_studio.repositories.serialization.dataset import DatasetRepository, sample_dataset
from xreport_studio.services.dataset_processing import DatasetProcessingService
from xreport_shared.services.jobs import JobManager
from xreport_studio.services.preparation import PreparationService
from xreport_studio.services.upload import UploadState
from xreport_shared.repositories.database.engine import Database
from xreport_inference.repositories.serialization.inference import InferenceRepository
from xreport_studio.repositories.serialization.validation import ValidationRepository


###############################################################################
@pytest.fixture
def database() -> Iterator[DatasetDatabase]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    database = DatasetDatabase.__new__(DatasetDatabase)
    database.engine = engine
    database.session = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    database.insert_batch_size = 100
    StudioBase.metadata.create_all(engine)
    yield database
    engine.dispose()


###############################################################################
def _preparation_service(database: DatasetDatabase) -> PreparationService:
    return PreparationService(
        repository=PreparationRepository(database),
        dataset_repository=cast(DatasetRepository, None),
        processing_service=cast(DatasetProcessingService, None),
        job_manager=cast(JobManager, None),
        upload_state=cast(UploadState, None),
        server_settings=cast(ServerSettings, SimpleNamespace()),
    )


###############################################################################
def test_source_dataset_delete_preserves_processed_samples_until_they_are_deleted(
    database: DatasetDatabase,
) -> None:
    now = datetime.now(timezone.utc)
    with database.transaction() as session:
        source = Dataset(name="source", name_key="source", created_at=now)
        processed = Dataset(name="processed", name_key="processed", created_at=now)
        session.add_all((source, processed))
        session.flush()

        record = DatasetRecord(
            dataset_id=source.dataset_id,
            image_name="image.png",
            image_name_key="image.png",
            image_path="C:/fixture/image.png",
            report_text="fixture report",
            row_order=1,
        )
        session.add(record)
        session.flush()
        run = ProcessingRun(
            dataset_id=processed.dataset_id,
            source_dataset_id=source.dataset_id,
            config_hash="fixture-config",
            executed_at=now,
            seed=42,
            sample_size=1.0,
            validation_size=0.2,
            split_seed=42,
            vocabulary_size=8,
            max_report_size=32,
            tokenizer="fixture-tokenizer",
        )
        session.add(run)
        session.flush()
        session.add(
            TrainingSample(
                processing_run_id=run.processing_run_id,
                record_id=record.record_id,
                split="train",
                tokens_json=[1, 2, 3],
            )
        )

    service = _preparation_service(database)
    app = FastAPI()
    register_service_error_handlers(app)
    router = APIRouter(prefix="/api/preparation")
    PreparationEndpoint(router, service).add_routes()
    app.include_router(router)

    with TestClient(app) as client:
        blocked = client.delete("/api/preparation/dataset/source")
        assert blocked.status_code == 409
        assert "processed" in blocked.json()["detail"]

        with database.read_session() as session:
            assert (
                session.execute(select(func.count()).select_from(Dataset)).scalar_one()
                == 2
            )
            assert (
                session.execute(
                    select(func.count()).select_from(DatasetRecord)
                ).scalar_one()
                == 1
            )
            assert (
                session.execute(
                    select(func.count()).select_from(TrainingSample)
                ).scalar_one()
                == 1
            )

        training_data, validation_data, _metadata = DatasetRepository(
            database
        ).load_training_data(dataset_name="processed")
        assert len(training_data) == 1
        assert validation_data.empty
        assert training_data.iloc[0]["image"] == "image.png"
        assert training_data.iloc[0]["text"] == "fixture report"
        assert training_data.iloc[0]["path"] == "C:/fixture/image.png"

        assert client.delete("/api/preparation/dataset/processed").status_code == 200
        assert client.delete("/api/preparation/dataset/source").status_code == 200

    with database.read_session() as session:
        assert (
            session.execute(select(func.count()).select_from(Dataset)).scalar_one() == 0
        )
        assert (
            session.execute(
                select(func.count()).select_from(DatasetRecord)
            ).scalar_one()
            == 0
        )
        assert (
            session.execute(
                select(func.count()).select_from(ProcessingRun)
            ).scalar_one()
            == 0
        )
        assert (
            session.execute(
                select(func.count()).select_from(TrainingSample)
            ).scalar_one()
            == 0
        )
        assert session.execute(text("PRAGMA foreign_key_check")).all() == []
        assert session.execute(text("PRAGMA integrity_check")).scalar_one() == "ok"


###############################################################################
def _memory_database(metadata) -> Database:
    database = Database.__new__(Database)
    database.engine = create_engine("sqlite:///:memory:", future=True)
    database.session = sessionmaker(
        bind=database.engine, future=True, expire_on_commit=False
    )
    database.insert_batch_size = 100
    metadata.create_all(database.engine)
    return database


###############################################################################
def _serializer() -> tuple[DatasetRepository, Database]:
    database = _memory_database(StudioBase.metadata)
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
def _seed_validation_dataset(
    repository: DatasetRepository, database: Database, name: str, texts: list[str]
) -> list[int]:
    repository.upsert_source_dataset(
        pd.DataFrame(
            [
                {
                    "dataset_name": name,
                    "image_name": f"{name}_{index}.png",
                    "report_text": text,
                    "image_path": f"{name}_{index}.png",
                }
                for index, text in enumerate(texts)
            ]
        )
    )
    with database.read_session() as session:
        return list(
            session.execute(
                select(DatasetRecord.record_id)
                .join(Dataset, Dataset.dataset_id == DatasetRecord.dataset_id)
                .where(Dataset.name == name)
                .order_by(DatasetRecord.row_order)
            ).scalars()
        )


###############################################################################
def _validation_run(name: str, record_ids: list[int], words: list[int], **extra) -> dict:
    return {
        "dataset_name": name,
        "sample_size": 1.0,
        "records_requested": len(record_ids),
        "records_evaluated": len(record_ids),
        "metrics": [{"metric": "text_statistics", "records_computed": len(words)}],
        "text_records": [
            {"record_id": record_id, "word_count": count, "char_count": count * 5,
             "unique_words": count}
            for record_id, count in zip(record_ids, words)
        ],
        "aggregates": [
            {"domain": "text", "measure": "words_per_report", "count": len(words),
             "mean": sum(words) / len(words), "minimum": float(min(words)),
             "maximum": float(max(words))}
        ],
        "histograms": {"report_length": [(0.0, 5.0, 1), (5.0, 10.0, len(words) - 1)]},
        **extra,
    }


###############################################################################
@pytest.mark.usefixtures("studio_profile")
def test_repeated_validation_runs_are_kept_as_history() -> None:
    dataset_repository, database = _serializer()
    ids = _seed_validation_dataset(dataset_repository, database, "Chest", ["a b", "c d e"])
    repository = ValidationRepository(database=database)
    first = repository.save_validation_run(_validation_run("Chest", ids, [2, 3]))
    second = repository.save_validation_run(
        _validation_run("Chest", ids[:1], [2], sample_size=0.5, seed=7)
    )

    assert second > first
    history = repository.list_validation_runs("Chest") or []
    assert [item["validation_run_id"] for item in history] == [second, first]
    latest = repository.get_validation_report("Chest") or {}
    assert latest["run"]["validation_run_id"] == second
    assert latest["run"]["seed"] == 7
    assert latest["run"]["records_evaluated"] == 1
    older = repository.get_validation_report("Chest", first) or {}
    assert older["aggregates"][0]["mean"] == 2.5
    assert [bin["count"] for bin in older["histograms"][0]["bins"]] == [1, 1]
    # Metrics that were not computed are absent, never reported as zero rows.
    assert [item["domain"] for item in older["aggregates"]] == ["text"]


###############################################################################
@pytest.mark.usefixtures("studio_profile")
def test_statistics_are_queryable_with_plain_sql_across_datasets() -> None:
    dataset_repository, database = _serializer()
    repository = ValidationRepository(database=database)
    for name, texts, words in (
        ("Chest", ["a", "b c"], [1, 2]),
        ("Spine", ["a b c d", "e f g h i j"], [4, 6]),
    ):
        ids = _seed_validation_dataset(dataset_repository, database, name, texts)
        repository.save_validation_run(_validation_run(name, ids, words))
    with database.read_session() as session:
        rows = session.execute(
            text(
                "SELECT d.name, AVG(t.word_count), MAX(t.word_count) "
                "FROM record_text_stats t "
                "JOIN validation_runs r ON r.validation_run_id = t.validation_run_id "
                "JOIN datasets d ON d.dataset_id = r.dataset_id "
                "GROUP BY d.name ORDER BY d.name"
            )
        ).all()
    assert rows == [("Chest", 1.5, 2), ("Spine", 5.0, 6)]


###############################################################################
@pytest.mark.usefixtures("studio_profile")
def test_record_statistics_page_sorts_filters_and_rejects_unknown_columns() -> None:
    dataset_repository, database = _serializer()
    ids = _seed_validation_dataset(
        dataset_repository, database, "Chest", ["a", "b c", "d e f"]
    )
    repository = ValidationRepository(database=database)
    run_id = repository.save_validation_run(_validation_run("Chest", ids, [1, 2, 3]))

    page = repository.list_run_records(
        "Chest", run_id, "text", sort="word_count", descending=True, limit=2
    )
    assert page is not None and page["total"] == 3
    assert [row["word_count"] for row in page["rows"]] == [3, 2]
    filtered = repository.list_run_records("Chest", run_id, "text", search="chest_1")
    assert filtered is not None and [row["image_name"] for row in filtered["rows"]] == [
        "Chest_1.png"
    ]
    assert repository.list_run_records("Chest", run_id, "image") == {"total": 0, "rows": []}
    assert repository.list_run_records("Chest", run_id + 1, "text") is None
    with pytest.raises(ValueError):
        repository.list_run_records("Chest", run_id, "text", sort="word_count; DROP")


###############################################################################
@pytest.mark.usefixtures("studio_profile")
def test_deleting_a_dataset_removes_its_statistics() -> None:
    dataset_repository, database = _serializer()
    with database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    ids = _seed_validation_dataset(dataset_repository, database, "Chest", ["a b"])
    ValidationRepository(database=database).save_validation_run(
        _validation_run("Chest", ids, [2])
    )
    with database.transaction() as session:
        session.delete(session.execute(select(Dataset)).scalar_one())
    with database.read_session() as session:
        for table in (
            "validation_runs",
            "validation_run_metrics",
            "record_text_stats",
            "validation_aggregates",
            "validation_histogram_bins",
        ):
            assert session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one() == 0


###############################################################################
@pytest.mark.usefixtures("studio_profile")
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
    database = _memory_database(InferenceBase.metadata)
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
    database = _memory_database(InferenceBase.metadata)
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


###############################################################################
@pytest.mark.usefixtures("studio_profile")
def test_validation_job_persists_measured_statistics(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2
    import numpy as np
    from unittest.mock import MagicMock
    from xreport_studio.services import validation_runs

    dataset_repository, database = _serializer()
    paths = []
    for index, (height, width) in enumerate([(8, 8), (16, 8)]):
        path = tmp_path / f"img_{index}.png"
        cv2.imwrite(str(path), np.full((height, width), 10 * (index + 1), np.uint8))
        paths.append(path)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    paths.append(corrupt)
    dataset_repository.upsert_source_dataset(
        pd.DataFrame(
            [
                {
                    "dataset_name": "Chest",
                    "image_name": path.name,
                    "report_text": text,
                    "image_path": str(path),
                }
                for path, text in zip(paths, ["one two", "one two three four", "x"])
            ]
        )
    )
    manager = MagicMock()
    manager.should_stop.return_value = False
    monkeypatch.setattr(validation_runs, "get_job_manager", lambda: manager)
    monkeypatch.setattr(validation_runs, "DatasetRepository", lambda: dataset_repository)
    monkeypatch.setattr(
        validation_runs, "ValidationRepository", lambda: ValidationRepository(database=database)
    )

    result = validation_runs.run_validation_job(
        {
            "dataset_name": "Chest",
            "sample_size": 1.0,
            "seed": 3,
            "metrics": ["text_statistics", "image_statistics", "pixels_distribution"],
        },
        "job-1",
    )

    report = ValidationRepository(database=database).get_validation_report(
        "Chest", result["validation_run_id"]
    )
    assert report is not None
    assert report["run"]["records_requested"] == 3
    coverage = {item["metric"]: item["records_computed"] for item in report["run"]["metrics"]}
    # The unreadable image is skipped, so image and pixel metrics cover 2 of 3 records.
    assert coverage == {
        "image_statistics": 2,
        "pixels_distribution": 2,
        "text_statistics": 3,
    }
    aggregates = {(item["domain"], item["measure"]): item for item in report["aggregates"]}
    assert aggregates[("text", "words_per_report")]["total"] == 7
    assert aggregates[("text", "words_per_report")]["median"] == 2
    assert aggregates[("image", "height")]["mean"] == 12.0
    assert aggregates[("image", "mean_intensity")]["maximum"] == 20.0
    histograms = {item["name"]: item["bins"] for item in report["histograms"]}
    assert sum(bin["count"] for bin in histograms["pixel_intensity"]) == 8 * 8 + 16 * 8
    assert sum(bin["count"] for bin in histograms["report_length"]) == 3
