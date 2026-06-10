from __future__ import annotations

from sqlalchemy import case, delete, exists, func, select

from server.repositories.database import XREPORTDatabase
from server.repositories.serialization.data import DataSerializer
from server.repositories.schemas import (
    Dataset,
    DatasetRecord,
    ProcessingRun,
    TrainingSample,
    ValidationRun,
)

###############################################################################
class PreparationRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: XREPORTDatabase) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def get_dataset_status(self) -> int:
        session = self.database.backend.session()
        try:
            return session.execute(select(func.count(DatasetRecord.record_id))).scalar() or 0
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def get_dataset_names(self):
        validation_exists = exists().where(ValidationRun.dataset_id == Dataset.dataset_id)
        stmt = (
            select(
                Dataset.name,
                func.min(DatasetRecord.image_path).label("sample_path"),
                func.count(DatasetRecord.record_id).label("row_count"),
                case((validation_exists, True), else_=False).label("has_validation_report"),
            )
            .join(DatasetRecord, DatasetRecord.dataset_id == Dataset.dataset_id)
            .group_by(Dataset.dataset_id, Dataset.name)
            .order_by(Dataset.name)
        )
        session = self.database.backend.session()
        try:
            return session.execute(stmt).all()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def get_processed_dataset_names(self):
        latest_runs = (
            select(
                ProcessingRun.dataset_id.label("dataset_id"),
                func.max(ProcessingRun.processing_run_id).label("processing_run_id"),
            )
            .group_by(ProcessingRun.dataset_id)
            .subquery()
        )
        validation_exists = exists().where(ValidationRun.dataset_id == Dataset.dataset_id)
        stmt = (
            select(
                Dataset.name,
                func.count(TrainingSample.training_sample_id).label("row_count"),
                case((validation_exists, True), else_=False).label("has_validation_report"),
            )
            .join(latest_runs, latest_runs.c.dataset_id == Dataset.dataset_id)
            .outerjoin(
                TrainingSample,
                TrainingSample.processing_run_id == latest_runs.c.processing_run_id,
            )
            .group_by(Dataset.dataset_id, Dataset.name)
            .order_by(Dataset.name)
        )
        session = self.database.backend.session()
        try:
            return session.execute(stmt).all()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def get_processing_metadata(self, dataset_name: str):
        serializer = DataSerializer()
        metadata = serializer.load_training_data(
            only_metadata=True,
            dataset_name=dataset_name,
        )
        if not isinstance(metadata, dict):
            return None
        return metadata

    # -------------------------------------------------------------------------
    def delete_dataset(self, dataset_name: str) -> int:
        session = self.database.backend.session()
        try:
            result = session.execute(delete(Dataset).where(Dataset.name == dataset_name))
            rowcount = int(result.rowcount or 0)
            if rowcount > 0:
                session.commit()
            return rowcount
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def get_dataset_by_name(self, dataset_name: str):
        session = self.database.backend.session()
        try:
            return session.execute(
                select(Dataset).where(Dataset.name == dataset_name)
            ).scalar_one_or_none()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def count_records(self, dataset_id: int) -> int:
        session = self.database.backend.session()
        try:
            return (
                session.execute(
                    select(func.count(DatasetRecord.record_id)).where(
                        DatasetRecord.dataset_id == dataset_id
                    )
                ).scalar()
                or 0
            )
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def get_record_at_index(self, dataset_id: int, index: int):
        session = self.database.backend.session()
        try:
            return session.execute(
                select(
                    DatasetRecord.image_name,
                    DatasetRecord.report_text,
                    DatasetRecord.image_path,
                )
                .where(DatasetRecord.dataset_id == dataset_id)
                .order_by(DatasetRecord.row_order, DatasetRecord.record_id)
                .offset(index)
                .limit(1)
            ).first()
        finally:
            session.close()
