from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

from XREPORT.server.repositories.schemas.types import JSONSequence

Base = declarative_base()


###############################################################################
class Dataset(Base):
    """Canonical dataset identity."""

    __tablename__ = "datasets"
    dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (UniqueConstraint("name", name="uq_datasets_name"),)
    records = relationship(
        "DatasetRecord",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    processing_runs = relationship(
        "ProcessingRun",
        back_populates="dataset",
        foreign_keys="ProcessingRun.dataset_id",
        cascade="all, delete-orphan",
    )
    source_processing_runs = relationship(
        "ProcessingRun",
        back_populates="source_dataset",
        foreign_keys="ProcessingRun.source_dataset_id",
    )
    validation_runs = relationship(
        "ValidationRun",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


###############################################################################
class DatasetRecord(Base):
    """Canonical image/report records belonging to a dataset."""

    __tablename__ = "dataset_records"
    record_id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
    )
    image_name = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    report_text = Column(String, nullable=False)
    row_order = Column(Integer, nullable=False)
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
    dataset = relationship("Dataset", back_populates="records")
    training_samples = relationship(
        "TrainingSample",
        back_populates="record",
        cascade="all, delete-orphan",
    )
    validation_image_stats = relationship(
        "ValidationImageStat",
        back_populates="record",
        cascade="all, delete-orphan",
    )
    inference_reports = relationship(
        "InferenceReport",
        back_populates="record",
    )


###############################################################################
class ProcessingRun(Base):
    """Preprocessing run metadata and configuration."""

    __tablename__ = "processing_runs"
    processing_run_id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
    )
    source_dataset_id = Column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="SET NULL"),
        nullable=True,
    )
    config_hash = Column(String, nullable=False)
    executed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    seed = Column(Integer, nullable=False)
    sample_size = Column(Float, nullable=False)
    validation_size = Column(Float, nullable=False)
    split_seed = Column(Integer, nullable=False)
    vocabulary_size = Column(Integer)
    max_report_size = Column(Integer, nullable=False)
    tokenizer = Column(String, nullable=False)
    __table_args__ = (
        Index("ix_processing_runs_config_hash", "config_hash"),
        Index("ix_processing_runs_dataset_id", "dataset_id"),
    )
    dataset = relationship(
        "Dataset",
        back_populates="processing_runs",
        foreign_keys=[dataset_id],
    )
    source_dataset = relationship(
        "Dataset",
        back_populates="source_processing_runs",
        foreign_keys=[source_dataset_id],
    )
    training_samples = relationship(
        "TrainingSample",
        back_populates="processing_run",
        cascade="all, delete-orphan",
    )


###############################################################################
class TrainingSample(Base):
    """Processed training samples linked to preprocessing runs and source records."""

    __tablename__ = "training_samples"
    training_sample_id = Column(Integer, primary_key=True, autoincrement=True)
    processing_run_id = Column(
        Integer,
        ForeignKey("processing_runs.processing_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    record_id = Column(
        Integer,
        ForeignKey("dataset_records.record_id", ondelete="CASCADE"),
        nullable=False,
    )
    split = Column(String, nullable=False)
    tokens_json = Column(JSONSequence, nullable=False)
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
    processing_run = relationship("ProcessingRun", back_populates="training_samples")
    record = relationship("DatasetRecord", back_populates="training_samples")


###############################################################################
class ValidationRun(Base):
    """Validation run metadata and aggregate metric payloads."""

    __tablename__ = "validation_runs"
    validation_run_id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
    )
    executed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    sample_size = Column(Float, nullable=False)
    metrics_json = Column(JSONSequence, nullable=False)
    artifacts_json = Column(JSONSequence)
    __table_args__ = (
        Index("ix_validation_runs_dataset_id", "dataset_id"),
        Index("ix_validation_runs_dataset_executed", "dataset_id", "executed_at"),
    )
    dataset = relationship("Dataset", back_populates="validation_runs")
    text_summary = relationship(
        "ValidationTextSummary",
        back_populates="validation_run",
        uselist=False,
        cascade="all, delete-orphan",
    )
    image_stats = relationship(
        "ValidationImageStat",
        back_populates="validation_run",
        cascade="all, delete-orphan",
    )
    pixel_distribution = relationship(
        "ValidationPixelDistribution",
        back_populates="validation_run",
        cascade="all, delete-orphan",
    )


###############################################################################
class ValidationTextSummary(Base):
    """Aggregate text statistics for a validation run."""

    __tablename__ = "validation_text_summary"
    validation_run_id = Column(
        Integer,
        ForeignKey("validation_runs.validation_run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    count = Column(Integer, nullable=False)
    total_words = Column(Integer, nullable=False)
    unique_words = Column(Integer, nullable=False)
    avg_words_per_report = Column(Float, nullable=False)
    min_words_per_report = Column(Integer, nullable=False)
    max_words_per_report = Column(Integer, nullable=False)
    validation_run = relationship("ValidationRun", back_populates="text_summary")


###############################################################################
class ValidationImageStat(Base):
    """Per-record image statistics for a validation run."""

    __tablename__ = "validation_image_stats"
    validation_run_id = Column(
        Integer,
        ForeignKey("validation_runs.validation_run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    record_id = Column(
        Integer,
        ForeignKey("dataset_records.record_id", ondelete="CASCADE"),
        primary_key=True,
    )
    height = Column(Integer)
    width = Column(Integer)
    mean = Column(Float)
    median = Column(Float)
    std = Column(Float)
    min = Column(Float)
    max = Column(Float)
    pixel_range = Column(Float)
    noise_std = Column(Float)
    noise_ratio = Column(Float)
    validation_run = relationship("ValidationRun", back_populates="image_stats")
    record = relationship("DatasetRecord", back_populates="validation_image_stats")


###############################################################################
class ValidationPixelDistribution(Base):
    """Pixel intensity distribution bins for a validation run."""

    __tablename__ = "validation_pixel_distribution"
    validation_run_id = Column(
        Integer,
        ForeignKey("validation_runs.validation_run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    bin = Column(Integer, primary_key=True)
    count = Column(Integer, nullable=False)
    __table_args__ = (
        CheckConstraint("bin >= 0 AND bin <= 255", name="ck_validation_pixel_bin"),
    )
    validation_run = relationship("ValidationRun", back_populates="pixel_distribution")


###############################################################################
class Checkpoint(Base):
    """Canonical checkpoint identity."""

    __tablename__ = "checkpoints"
    checkpoint_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (
        UniqueConstraint("name", name="uq_checkpoints_name"),
        UniqueConstraint("path", name="uq_checkpoints_path"),
    )
    evaluations = relationship(
        "CheckpointEvaluation",
        back_populates="checkpoint",
        cascade="all, delete-orphan",
    )
    inference_runs = relationship(
        "InferenceRun",
        back_populates="checkpoint",
        cascade="all, delete-orphan",
    )


###############################################################################
class CheckpointEvaluation(Base):
    """Latest checkpoint evaluation payload."""

    __tablename__ = "checkpoint_evaluations"
    evaluation_id = Column(Integer, primary_key=True, autoincrement=True)
    checkpoint_id = Column(
        Integer,
        ForeignKey("checkpoints.checkpoint_id", ondelete="CASCADE"),
        nullable=False,
    )
    executed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    metrics_json = Column(JSONSequence, nullable=False)
    metric_configs_json = Column(JSONSequence, nullable=False)
    results_json = Column(JSONSequence, nullable=False)
    __table_args__ = (
        Index("ix_checkpoint_evaluations_checkpoint_id", "checkpoint_id"),
    )
    checkpoint = relationship("Checkpoint", back_populates="evaluations")


###############################################################################
class InferenceRun(Base):
    """Inference execution metadata."""

    __tablename__ = "inference_runs"
    inference_run_id = Column(Integer, primary_key=True, autoincrement=True)
    checkpoint_id = Column(
        Integer,
        ForeignKey("checkpoints.checkpoint_id", ondelete="CASCADE"),
        nullable=False,
    )
    generation_mode = Column(String, nullable=False)
    request_id = Column(String)
    executed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (
        UniqueConstraint("request_id", name="uq_inference_runs_request_id"),
    )
    checkpoint = relationship("Checkpoint", back_populates="inference_runs")
    reports = relationship(
        "InferenceReport",
        back_populates="inference_run",
        cascade="all, delete-orphan",
    )


###############################################################################
class InferenceReport(Base):
    """Generated reports linked to inference runs."""

    __tablename__ = "inference_reports"
    inference_report_id = Column(Integer, primary_key=True, autoincrement=True)
    inference_run_id = Column(
        Integer,
        ForeignKey("inference_runs.inference_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    record_id = Column(
        Integer,
        ForeignKey("dataset_records.record_id", ondelete="SET NULL"),
    )
    input_image_name = Column(String, nullable=False)
    generated_report = Column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "inference_run_id",
            "input_image_name",
            name="uq_inference_reports_run_image",
        ),
    )
    inference_run = relationship("InferenceRun", back_populates="reports")
    record = relationship("DatasetRecord", back_populates="inference_reports")
