from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from XREPORT.server.repositories.schemas.types import JSONSequence


class Base(DeclarativeBase):
    pass


###############################################################################
class Dataset(Base):
    """Canonical dataset identity."""

    __tablename__ = "datasets"
    dataset_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (UniqueConstraint("name", name="uq_datasets_name"),)
    records: Mapped[list[DatasetRecord]] = relationship(
        "DatasetRecord",
        back_populates="dataset",
        cascade="all, delete-orphan",
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
    image_name: Mapped[str] = mapped_column(String, nullable=False)
    image_path: Mapped[str] = mapped_column(String, nullable=False)
    report_text: Mapped[str] = mapped_column(String, nullable=False)
    row_order: Mapped[int] = mapped_column(Integer, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "image_name",
            "report_text",
            name="uq_dataset_records_dataset_image_report",
        ),
        Index("ix_dataset_records_dataset_order", "dataset_id", "row_order"),
        Index("ix_dataset_records_dataset_image", "dataset_id", "image_name"),
    )
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="records")
    training_samples: Mapped[list[TrainingSample]] = relationship(
        "TrainingSample",
        back_populates="record",
        cascade="all, delete-orphan",
    )
    validation_image_stats: Mapped[list[ValidationImageStat]] = relationship(
        "ValidationImageStat",
        back_populates="record",
        cascade="all, delete-orphan",
    )
    inference_reports: Mapped[list[InferenceReport]] = relationship(
        "InferenceReport",
        back_populates="record",
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
        DateTime(timezone=True),
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
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
    )
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    sample_size: Mapped[float] = mapped_column(Float, nullable=False)
    metrics_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    artifacts_json: Mapped[Any | None] = mapped_column(JSONSequence)
    __table_args__ = (
        Index("ix_validation_runs_dataset_id", "dataset_id"),
        Index("ix_validation_runs_dataset_executed", "dataset_id", "executed_at"),
    )
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="validation_runs")
    text_summary: Mapped[ValidationTextSummary | None] = relationship(
        "ValidationTextSummary",
        back_populates="validation_run",
        uselist=False,
        cascade="all, delete-orphan",
    )
    image_stats: Mapped[list[ValidationImageStat]] = relationship(
        "ValidationImageStat",
        back_populates="validation_run",
        cascade="all, delete-orphan",
    )
    pixel_distribution: Mapped[list[ValidationPixelDistribution]] = relationship(
        "ValidationPixelDistribution",
        back_populates="validation_run",
        cascade="all, delete-orphan",
    )


###############################################################################
class ValidationTextSummary(Base):
    """Aggregate text statistics for a validation run."""

    __tablename__ = "validation_text_summary"
    validation_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("validation_runs.validation_run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_words: Mapped[int] = mapped_column(Integer, nullable=False)
    unique_words: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_words_per_report: Mapped[float] = mapped_column(Float, nullable=False)
    min_words_per_report: Mapped[int] = mapped_column(Integer, nullable=False)
    max_words_per_report: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_run: Mapped[ValidationRun] = relationship(
        "ValidationRun", back_populates="text_summary"
    )


###############################################################################
class ValidationImageStat(Base):
    """Per-record image statistics for a validation run."""

    __tablename__ = "validation_image_stats"
    validation_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("validation_runs.validation_run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    record_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("dataset_records.record_id", ondelete="CASCADE"),
        primary_key=True,
    )
    height: Mapped[int | None] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer)
    mean: Mapped[float | None] = mapped_column(Float)
    median: Mapped[float | None] = mapped_column(Float)
    std: Mapped[float | None] = mapped_column(Float)
    min: Mapped[float | None] = mapped_column(Float)
    max: Mapped[float | None] = mapped_column(Float)
    pixel_range: Mapped[float | None] = mapped_column(Float)
    noise_std: Mapped[float | None] = mapped_column(Float)
    noise_ratio: Mapped[float | None] = mapped_column(Float)
    validation_run: Mapped[ValidationRun] = relationship(
        "ValidationRun", back_populates="image_stats"
    )
    record: Mapped[DatasetRecord] = relationship(
        "DatasetRecord", back_populates="validation_image_stats"
    )


###############################################################################
class ValidationPixelDistribution(Base):
    """Pixel intensity distribution bins for a validation run."""

    __tablename__ = "validation_pixel_distribution"
    validation_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("validation_runs.validation_run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    bin: Mapped[int] = mapped_column(Integer, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    __table_args__ = (
        CheckConstraint("bin >= 0 AND bin <= 255", name="ck_validation_pixel_bin"),
    )
    validation_run: Mapped[ValidationRun] = relationship(
        "ValidationRun", back_populates="pixel_distribution"
    )


###############################################################################
class Checkpoint(Base):
    """Canonical checkpoint identity."""

    __tablename__ = "checkpoints"
    checkpoint_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (
        UniqueConstraint("name", name="uq_checkpoints_name"),
        UniqueConstraint("path", name="uq_checkpoints_path"),
    )
    evaluations: Mapped[list[CheckpointEvaluation]] = relationship(
        "CheckpointEvaluation",
        back_populates="checkpoint",
        cascade="all, delete-orphan",
    )
    inference_runs: Mapped[list[InferenceRun]] = relationship(
        "InferenceRun",
        back_populates="checkpoint",
        cascade="all, delete-orphan",
    )


###############################################################################
class CheckpointEvaluation(Base):
    """Latest checkpoint evaluation payload."""

    __tablename__ = "checkpoint_evaluations"
    evaluation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    checkpoint_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("checkpoints.checkpoint_id", ondelete="CASCADE"),
        nullable=False,
    )
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    metrics_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    metric_configs_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    results_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    __table_args__ = (
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
        ForeignKey("checkpoints.checkpoint_id", ondelete="CASCADE"),
        nullable=False,
    )
    generation_mode: Mapped[str] = mapped_column(String, nullable=False)
    request_id: Mapped[str | None] = mapped_column(String)
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (
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
    generated_report: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "inference_run_id",
            "input_image_name",
            name="uq_inference_reports_run_image",
        ),
    )
    inference_run: Mapped[InferenceRun] = relationship("InferenceRun", back_populates="reports")
    record: Mapped[DatasetRecord | None] = relationship(
        "DatasetRecord", back_populates="inference_reports"
    )
