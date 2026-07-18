from __future__ import annotations

from sqlalchemy import case, delete, exists, func, select

from server.repositories.database import Database
from server.repositories.serialization.dataset import DatasetRepository
from server.repositories.schemas import (
    Dataset,
    DatasetRecord,
    DatasetVersion,
    ProcessingRun,
    TrainingSample,
    ValidationRun,
)
from server.repositories.schemas.normalization import normalize_key

###############################################################################
class PreparationRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: Database) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def get_dataset_status(self) -> int:
        session = self.database.session()
        try:
            return session.execute(select(func.count(DatasetRecord.record_id))).scalar() or 0
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def get_dataset_names(self):
        latest_versions = (
            select(
                DatasetVersion.dataset_id,
                func.max(DatasetVersion.version_number).label("version_number"),
            )
            .group_by(DatasetVersion.dataset_id)
            .subquery()
        )
        validation_exists = exists().where(ValidationRun.dataset_id == Dataset.dataset_id)
        stmt = (
            select(
                Dataset.name,
                func.min(DatasetRecord.image_path).label("sample_path"),
                func.count(DatasetRecord.record_id).label("row_count"),
                case((validation_exists, True), else_=False).label("has_validation_report"),
            )
            .join(DatasetRecord, DatasetRecord.dataset_id == Dataset.dataset_id)
            .join(
                DatasetVersion,
                DatasetVersion.dataset_version_id == DatasetRecord.dataset_version_id,
            )
            .join(
                latest_versions,
                (latest_versions.c.dataset_id == DatasetVersion.dataset_id)
                & (latest_versions.c.version_number == DatasetVersion.version_number),
            )
            .group_by(Dataset.dataset_id, Dataset.name)
            .order_by(Dataset.name)
        )
        session = self.database.session()
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
        session = self.database.session()
        try:
            return session.execute(stmt).all()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def get_processing_metadata(self, dataset_name: str):
        serializer = DatasetRepository()
        metadata = serializer.load_training_data(
            only_metadata=True,
            dataset_name=dataset_name,
        )
        if not isinstance(metadata, dict):
            return None
        return metadata

    # -------------------------------------------------------------------------
    def delete_dataset(self, dataset_name: str) -> int:
        session = self.database.session()
        try:
            result = session.execute(
                delete(Dataset).where(Dataset.name_key == normalize_key(dataset_name))
            )
            rowcount = int(getattr(result, "rowcount", 0) or 0)
            if rowcount > 0:
                session.commit()
            return rowcount
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def get_dataset_by_name(self, dataset_name: str):
        session = self.database.session()
        try:
            return session.execute(
                select(Dataset).where(Dataset.name_key == normalize_key(dataset_name))
            ).scalar_one_or_none()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def count_records(self, dataset_id: int) -> int:
        latest_versions = (
            select(
                DatasetVersion.dataset_id,
                func.max(DatasetVersion.version_number).label("version_number"),
            )
            .where(DatasetVersion.dataset_id == dataset_id)
            .group_by(DatasetVersion.dataset_id)
            .subquery()
        )
        session = self.database.session()
        try:
            return (
                session.execute(
                    select(func.count(DatasetRecord.record_id))
                    .join(
                        DatasetVersion,
                        DatasetVersion.dataset_version_id
                        == DatasetRecord.dataset_version_id,
                    )
                    .where(
                        DatasetRecord.dataset_id == dataset_id,
                        DatasetVersion.version_number == latest_versions.c.version_number,
                    )
                ).scalar()
                or 0
            )
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def get_record_at_index(self, dataset_id: int, index: int):
        latest_versions = (
            select(
                DatasetVersion.dataset_id,
                func.max(DatasetVersion.version_number).label("version_number"),
            )
            .where(DatasetVersion.dataset_id == dataset_id)
            .group_by(DatasetVersion.dataset_id)
            .subquery()
        )
        session = self.database.session()
        try:
            return session.execute(
                select(
                    DatasetRecord.image_name,
                    DatasetRecord.report_text,
                    DatasetRecord.image_path,
                )
                .where(DatasetRecord.dataset_id == dataset_id)
                .join(
                    DatasetVersion,
                    DatasetVersion.dataset_version_id == DatasetRecord.dataset_version_id,
                )
                .where(
                    DatasetVersion.version_number == latest_versions.c.version_number,
                    DatasetRecord.row_order == index + 1,
                )
                .order_by(DatasetRecord.row_order, DatasetRecord.record_id)
                .limit(1)
            ).first()
        finally:
            session.close()
