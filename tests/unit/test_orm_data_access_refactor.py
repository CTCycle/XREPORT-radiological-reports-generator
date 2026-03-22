from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from XREPORT.server.configurations.server import server_settings
from XREPORT.server.repositories.schemas import (
    Base,
    Checkpoint,
    CheckpointEvaluation,
    Dataset,
    DatasetRecord,
    ProcessingRun,
    TrainingSample,
    ValidationImageStat,
    ValidationPixelDistribution,
    ValidationRun,
    ValidationTextSummary,
)
from XREPORT.server.repositories.serialization.data import DataSerializer
from XREPORT.server.routes.preparation import PreparationEndpoint


###############################################################################
class BackendStub:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    # -------------------------------------------------------------------------
    def session(self):
        return self.session_factory()


###############################################################################
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


###############################################################################
@pytest.fixture()
def session_factory(tmp_path: Path):
    db_path = tmp_path / "orm_refactor_test.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, future=True)
    try:
        yield SessionLocal
    finally:
        engine.dispose()


###############################################################################
def create_preparation_endpoint(session_factory: sessionmaker) -> PreparationEndpoint:
    backend = BackendStub(session_factory)
    database_stub = SimpleNamespace(backend=backend)
    endpoint = PreparationEndpoint(
        router=APIRouter(),
        database=database_stub,
        job_manager=SimpleNamespace(),
        upload_state=SimpleNamespace(),
        server_settings=server_settings,
    )
    endpoint.allow_local_filesystem_access = True
    return endpoint


###############################################################################
def create_processing_run(
    session,
    dataset_id: int,
    config_hash: str,
    source_dataset_id: int | None = None,
) -> ProcessingRun:
    run = ProcessingRun(
        dataset_id=dataset_id,
        source_dataset_id=source_dataset_id,
        config_hash=config_hash,
        executed_at=utc_now(),
        seed=42,
        sample_size=1.0,
        validation_size=0.2,
        split_seed=42,
        vocabulary_size=128,
        max_report_size=256,
        tokenizer="tokenizer",
    )
    session.add(run)
    session.flush()
    return run


###############################################################################
def test_preparation_dataset_names_and_flags(session_factory: sessionmaker, tmp_path: Path):
    image_a = tmp_path / "sample_a.png"
    image_a.write_bytes(b"a")
    image_b = tmp_path / "sample_b.png"
    image_b.write_bytes(b"b")

    session = session_factory()
    try:
        dataset_with_report = Dataset(name="dataset_a", created_at=utc_now())
        dataset_without_report = Dataset(name="dataset_b", created_at=utc_now())
        session.add_all([dataset_with_report, dataset_without_report])
        session.flush()

        session.add_all(
            [
                DatasetRecord(
                    dataset_id=dataset_with_report.dataset_id,
                    image_name="a_1.png",
                    report_text="Report A1",
                    image_path=str(image_a),
                    row_order=1,
                ),
                DatasetRecord(
                    dataset_id=dataset_with_report.dataset_id,
                    image_name="a_2.png",
                    report_text="Report A2",
                    image_path=str(image_b),
                    row_order=2,
                ),
                DatasetRecord(
                    dataset_id=dataset_without_report.dataset_id,
                    image_name="b_1.png",
                    report_text="Report B1",
                    image_path=str(image_b),
                    row_order=1,
                ),
            ]
        )
        session.add(
            ValidationRun(
                dataset_id=dataset_with_report.dataset_id,
                executed_at=utc_now(),
                sample_size=1.0,
                metrics_json=[],
                artifacts_json={},
            )
        )
        session.commit()
    finally:
        session.close()

    endpoint = create_preparation_endpoint(session_factory)
    response = endpoint.get_dataset_names()
    by_name = {item.name: item for item in response.datasets}

    assert response.count == 2
    assert by_name["dataset_a"].row_count == 2
    assert by_name["dataset_a"].has_validation_report is True
    assert by_name["dataset_b"].row_count == 1
    assert by_name["dataset_b"].has_validation_report is False


###############################################################################
def test_preparation_processed_names_uses_latest_run(session_factory: sessionmaker):
    session = session_factory()
    try:
        dataset = Dataset(name="processed_dataset", created_at=utc_now())
        session.add(dataset)
        session.flush()

        old_run = create_processing_run(
            session,
            dataset_id=dataset.dataset_id,
            config_hash="old_hash",
        )
        new_run = create_processing_run(
            session,
            dataset_id=dataset.dataset_id,
            config_hash="new_hash",
        )

        record = DatasetRecord(
            dataset_id=dataset.dataset_id,
            image_name="sample.png",
            report_text="Report",
            image_path="sample.png",
            row_order=1,
        )
        session.add(record)
        session.flush()

        session.add_all(
            [
                TrainingSample(
                    processing_run_id=old_run.processing_run_id,
                    record_id=record.record_id,
                    split="train",
                    tokens_json=[1, 2],
                ),
                TrainingSample(
                    processing_run_id=new_run.processing_run_id,
                    record_id=record.record_id,
                    split="validation",
                    tokens_json=[3, 4],
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    endpoint = create_preparation_endpoint(session_factory)
    response = endpoint.get_processed_dataset_names()

    assert response.count == 1
    assert response.datasets[0].name == "processed_dataset"
    assert response.datasets[0].row_count == 1


###############################################################################
def test_data_serializer_get_validation_report_aggregates(session_factory: sessionmaker):
    session = session_factory()
    try:
        dataset = Dataset(name="validation_dataset", created_at=utc_now())
        session.add(dataset)
        session.flush()

        record = DatasetRecord(
            dataset_id=dataset.dataset_id,
            image_name="img.png",
            report_text="text",
            image_path="img.png",
            row_order=1,
        )
        session.add(record)
        session.flush()

        run = ValidationRun(
            dataset_id=dataset.dataset_id,
            executed_at=utc_now(),
            sample_size=0.5,
            metrics_json=[{"metric": "bleu", "value": 0.9}],
            artifacts_json={"artifact": "ok"},
        )
        session.add(run)
        session.flush()

        session.add(
            ValidationTextSummary(
                validation_run_id=run.validation_run_id,
                count=1,
                total_words=10,
                unique_words=7,
                avg_words_per_report=10.0,
                min_words_per_report=10,
                max_words_per_report=10,
            )
        )
        session.add(
            ValidationImageStat(
                validation_run_id=run.validation_run_id,
                record_id=record.record_id,
                height=224,
                width=224,
                mean=0.5,
                median=0.5,
                std=0.1,
                min=0.0,
                max=1.0,
                pixel_range=1.0,
                noise_std=0.02,
                noise_ratio=0.1,
            )
        )
        session.add_all(
            [
                ValidationPixelDistribution(
                    validation_run_id=run.validation_run_id,
                    bin=0,
                    count=3,
                ),
                ValidationPixelDistribution(
                    validation_run_id=run.validation_run_id,
                    bin=1,
                    count=5,
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    serializer = DataSerializer(queries=SimpleNamespace(backend=BackendStub(session_factory)))
    report = serializer.get_validation_report("validation_dataset")

    assert report is not None
    assert report["dataset_name"] == "validation_dataset"
    assert report["sample_size"] == 0.5
    assert report["text_statistics"]["count"] == 1
    assert report["pixel_distribution"]["bins"] == [0, 1]
    assert report["pixel_distribution"]["counts"] == [3, 5]


###############################################################################
def test_data_serializer_checkpoint_evaluation_exists_orm(session_factory: sessionmaker):
    session = session_factory()
    try:
        checkpoint = Checkpoint(
            name="checkpoint_a",
            path="checkpoint_a",
            created_at=utc_now(),
        )
        session.add(checkpoint)
        session.flush()
        session.add(
            CheckpointEvaluation(
                checkpoint_id=checkpoint.checkpoint_id,
                executed_at=utc_now(),
                metrics_json=[{"metric": "bleu"}],
                metric_configs_json={},
                results_json={},
            )
        )
        session.commit()
    finally:
        session.close()

    serializer = DataSerializer(queries=SimpleNamespace(backend=BackendStub(session_factory)))
    assert serializer.checkpoint_evaluation_report_exists("checkpoint_a") is True
    assert serializer.checkpoint_evaluation_report_exists("missing") is False
