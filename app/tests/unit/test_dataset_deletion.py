from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.api.errors import register_service_error_handlers
from server.api.preparation import PreparationEndpoint
from server.configurations import ServerSettings
from server.repositories.database import Database
from server.repositories.preparation import PreparationRepository
from server.repositories.schemas import (
    Base,
    Dataset,
    DatasetRecord,
    ProcessingRun,
    TrainingSample,
)
from server.repositories.serialization.dataset import DatasetRepository
from server.services.dataset_processing import DatasetProcessingService
from server.services.jobs import JobManager
from server.services.preparation import PreparationService
from server.services.upload import UploadState


@pytest.fixture
def database() -> Iterator[Database]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    database = Database.__new__(Database)
    database.engine = engine
    database.session = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    database.insert_batch_size = 100
    Base.metadata.create_all(engine)
    yield database
    engine.dispose()


def _preparation_service(database: Database) -> PreparationService:
    return PreparationService(
        repository=PreparationRepository(database),
        dataset_repository=cast(DatasetRepository, None),
        processing_service=cast(DatasetProcessingService, None),
        job_manager=cast(JobManager, None),
        upload_state=cast(UploadState, None),
        server_settings=cast(
            ServerSettings,
            SimpleNamespace(
                features=SimpleNamespace(allow_local_filesystem_access=False)
            ),
        ),
    )


def test_source_dataset_delete_preserves_processed_samples_until_they_are_deleted(
    database: Database,
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
