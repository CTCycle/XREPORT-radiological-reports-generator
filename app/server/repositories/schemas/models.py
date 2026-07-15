from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from server.repositories.schemas.normalization import normalize_key
from server.repositories.schemas.types import JSONSequence, UTCDateTime

###############################################################################
class Base(DeclarativeBase):
    pass

###############################################################################
class Dataset(Base):
    """Canonical dataset identity."""

    __tablename__ = "datasets"
    dataset_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    name_key: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (UniqueConstraint("name_key", name="uq_datasets_name_key"),)
    records: Mapped[list[DatasetRecord]] = relationship(
        "DatasetRecord",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    versions: Mapped[list[DatasetVersion]] = relationship(
        "DatasetVersion",
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="DatasetVersion.version_number",
    )
    processing_runs: Mapped[list[ProcessingRun]] = relationship(
        "ProcessingRun",
        back_populates="dataset",
        foreign_keys="ProcessingRun.dataset_id",
        cascade="all, delete-orphan",
    )
    source_processing_runs: Mapped[list[ProcessingRun]] = relationship(
        "ProcessingRun",
        back_populates="source_dataset",
        foreign_keys="ProcessingRun.source_dataset_id",
    )
    validation_runs: Mapped[list[ValidationRun]] = relationship(
        "ValidationRun",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )

###############################################################################
class DatasetRecord(Base):
    """Canonical image/report records belonging to a dataset."""

    __tablename__ = "dataset_records"
    record_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_version_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("dataset_versions.dataset_version_id", ondelete="CASCADE"),
        nullable=True,
    )
    image_name: Mapped[str] = mapped_column(String(512), nullable=False)
    image_name_key: Mapped[str] = mapped_column(String(512), nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    report_text: Mapped[str] = mapped_column(Text, nullable=False)
    row_order: Mapped[int] = mapped_column(Integer, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "dataset_version_id",
            "image_name_key",
            name="uq_dataset_records_dataset_image_key",
        ),
        UniqueConstraint(
            "dataset_version_id",
            "row_order",
            name="uq_dataset_records_dataset_order",
        ),
        Index("ix_dataset_records_dataset_order", "dataset_id", "row_order"),
        Index("ix_dataset_records_dataset_image", "dataset_id", "image_name"),
    )
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="records")
    dataset_version: Mapped[DatasetVersion | None] = relationship(
        "DatasetVersion", back_populates="records"
    )
    training_samples: Mapped[list[TrainingSample]] = relationship(
        "TrainingSample",
        back_populates="record",
        cascade="all, delete-orphan",
    )
    inference_reports: Mapped[list[InferenceReport]] = relationship(
        "InferenceReport",
        back_populates="record",
    )


class DatasetVersion(Base):
    """Immutable snapshot of one logical imported dataset."""

    __tablename__ = "dataset_versions"
    dataset_version_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    imported_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    __table_args__ = (
        UniqueConstraint("dataset_id", "version_number", name="uq_dataset_versions_number"),
        UniqueConstraint("dataset_id", "content_hash", name="uq_dataset_versions_hash"),
        Index("ix_dataset_versions_dataset_latest", "dataset_id", "version_number"),
        CheckConstraint("version_number > 0", name="ck_dataset_versions_number"),
        CheckConstraint("record_count >= 0", name="ck_dataset_versions_record_count"),
    )
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="versions")
    records: Mapped[list[DatasetRecord]] = relationship(
        "DatasetRecord", back_populates="dataset_version", cascade="all, delete-orphan"
    )

###############################################################################
class ProcessingRun(Base):
    """Preprocessing run metadata and configuration."""

    __tablename__ = "processing_runs"
    processing_run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
    )
    source_dataset_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="SET NULL"),
        nullable=True,
    )
    config_hash: Mapped[str] = mapped_column(String, nullable=False)
    executed_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_size: Mapped[float] = mapped_column(Float, nullable=False)
    validation_size: Mapped[float] = mapped_column(Float, nullable=False)
    split_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    vocabulary_size: Mapped[int | None] = mapped_column(Integer)
    max_report_size: Mapped[int] = mapped_column(Integer, nullable=False)
    tokenizer: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        Index("ix_processing_runs_config_hash", "config_hash"),
        Index("ix_processing_runs_dataset_id", "dataset_id"),
    )
    dataset: Mapped[Dataset] = relationship(
        "Dataset",
        back_populates="processing_runs",
        foreign_keys=[dataset_id],
    )
    source_dataset: Mapped[Dataset | None] = relationship(
        "Dataset",
        back_populates="source_processing_runs",
        foreign_keys=[source_dataset_id],
    )
    training_samples: Mapped[list[TrainingSample]] = relationship(
        "TrainingSample",
        back_populates="processing_run",
        cascade="all, delete-orphan",
    )

###############################################################################
class TrainingSample(Base):
    """Processed training samples linked to preprocessing runs and source records."""

    __tablename__ = "training_samples"
    training_sample_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    processing_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("processing_runs.processing_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    record_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("dataset_records.record_id", ondelete="CASCADE"),
        nullable=False,
    )
    split: Mapped[str] = mapped_column(String, nullable=False)
    tokens_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "processing_run_id",
            "record_id",
            name="uq_training_samples_run_record",
        ),
        CheckConstraint(
            "split IN ('train', 'validation')",
            name="ck_training_samples_split",
        ),
        Index("ix_training_samples_run_split", "processing_run_id", "split"),
    )
    processing_run: Mapped[ProcessingRun] = relationship(
        "ProcessingRun", back_populates="training_samples"
    )
    record: Mapped[DatasetRecord] = relationship("DatasetRecord", back_populates="training_samples")

###############################################################################
class ValidationRun(Base):
    """Validation run metadata and aggregate metric payloads."""

    __tablename__ = "validation_runs"
    validation_run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="succeeded")
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
    )
    executed_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    sample_size: Mapped[float] = mapped_column(Float, nullable=False)
    metrics_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    artifacts_json: Mapped[Any | None] = mapped_column(JSONSequence)
    text_count: Mapped[int | None] = mapped_column(Integer)
    text_total_words: Mapped[int | None] = mapped_column(Integer)
    text_unique_words: Mapped[int | None] = mapped_column(Integer)
    text_avg_words: Mapped[float | None] = mapped_column(Float)
    text_min_words: Mapped[int | None] = mapped_column(Integer)
    text_max_words: Mapped[int | None] = mapped_column(Integer)
    image_count: Mapped[int | None] = mapped_column(Integer)
    image_mean_height: Mapped[float | None] = mapped_column(Float)
    image_mean_width: Mapped[float | None] = mapped_column(Float)
    image_mean_value: Mapped[float | None] = mapped_column(Float)
    image_std_value: Mapped[float | None] = mapped_column(Float)
    image_mean_noise_std: Mapped[float | None] = mapped_column(Float)
    image_mean_noise_ratio: Mapped[float | None] = mapped_column(Float)
    pixel_bins_json: Mapped[Any | None] = mapped_column(JSONSequence)
    pixel_counts_json: Mapped[Any | None] = mapped_column(JSONSequence)
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_validation_runs_status",
        ),
        Index("ix_validation_runs_dataset_id", "dataset_id"),
        Index("ix_validation_runs_dataset_executed", "dataset_id", "executed_at"),
    )
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="validation_runs")
###############################################################################
class Checkpoint(Base):
    """Canonical checkpoint identity."""

    __tablename__ = "checkpoints"
    checkpoint_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    name_key: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    __table_args__ = (
        UniqueConstraint("name_key", name="uq_checkpoints_name_key"),
    )
    evaluations: Mapped[list[CheckpointEvaluation]] = relationship(
        "CheckpointEvaluation",
        back_populates="checkpoint",
        passive_deletes=True,
    )
    inference_runs: Mapped[list[InferenceRun]] = relationship(
        "InferenceRun",
        back_populates="checkpoint",
        passive_deletes=True,
    )

###############################################################################
class CheckpointEvaluation(Base):
    """Latest checkpoint evaluation payload."""

    __tablename__ = "checkpoint_evaluations"
    evaluation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="succeeded")
    checkpoint_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("checkpoints.checkpoint_id", ondelete="RESTRICT"),
        nullable=False,
    )
    executed_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    metrics_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    metric_configs_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    results_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_checkpoint_evaluations_status",
        ),
        Index("ix_checkpoint_evaluations_checkpoint_id", "checkpoint_id"),
    )
    checkpoint: Mapped[Checkpoint] = relationship("Checkpoint", back_populates="evaluations")

###############################################################################
class InferenceRun(Base):
    """Inference execution metadata."""

    __tablename__ = "inference_runs"
    inference_run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    checkpoint_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("checkpoints.checkpoint_id", ondelete="RESTRICT"),
        nullable=False,
    )
    generation_mode: Mapped[str] = mapped_column(String, nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="succeeded")
    executed_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_inference_runs_status",
        ),
        UniqueConstraint("request_id", name="uq_inference_runs_request_id"),
    )
    checkpoint: Mapped[Checkpoint] = relationship("Checkpoint", back_populates="inference_runs")
    reports: Mapped[list[InferenceReport]] = relationship(
        "InferenceReport",
        back_populates="inference_run",
        cascade="all, delete-orphan",
    )

###############################################################################
class InferenceReport(Base):
    """Generated reports linked to inference runs."""

    __tablename__ = "inference_reports"
    inference_report_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inference_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("inference_runs.inference_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    record_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("dataset_records.record_id", ondelete="SET NULL"),
        nullable=True,
    )
    input_image_name: Mapped[str] = mapped_column(String, nullable=False)
    input_image_name_key: Mapped[str] = mapped_column(String(512), nullable=False)
    image_index: Mapped[int] = mapped_column(Integer, nullable=False)
    generated_report: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "inference_run_id",
            "input_image_name_key",
            name="uq_inference_reports_run_image",
        ),
        UniqueConstraint(
            "inference_run_id", "image_index", name="uq_inference_reports_run_index"
        ),
        CheckConstraint("image_index >= 0", name="ck_inference_reports_image_index"),
    )
    inference_run: Mapped[InferenceRun] = relationship("InferenceRun", back_populates="reports")
    record: Mapped[DatasetRecord | None] = relationship(
        "DatasetRecord", back_populates="inference_reports"
    )


@event.listens_for(Dataset, "before_insert")
def _populate_dataset_name_key(_mapper: Any, _connection: Any, target: Dataset) -> None:
    target.name_key = normalize_key(target.name)


@event.listens_for(DatasetRecord, "before_insert")
def _populate_image_name_key(
    _mapper: Any, _connection: Any, target: DatasetRecord
) -> None:
    target.image_name_key = normalize_key(target.image_name)


@event.listens_for(Checkpoint, "before_insert")
def _populate_checkpoint_name_key(
    _mapper: Any, _connection: Any, target: Checkpoint
) -> None:
    target.name_key = normalize_key(target.name)
