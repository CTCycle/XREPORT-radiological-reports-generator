"""XREPORT Studio ORM schema: datasets, preparation, training, and evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
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

from xreport_shared.repositories.schemas.checkpoint import CheckpointColumns
from xreport_shared.repositories.schemas.normalization import normalize_key
from xreport_shared.repositories.schemas.types import JSONSequence, UTCDateTime

###############################################################################
class StudioBase(DeclarativeBase):
    pass

###############################################################################
class StudioSettingsRecord(StudioBase):
    """Singleton persisted XREPORT Studio settings row."""

    __tablename__ = "application_settings"
    settings_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    global_seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    job_polling_interval: Mapped[float] = mapped_column(Float, nullable=False)
    theme_preference: Mapped[str | None] = mapped_column(
        String(8), nullable=True, default=None
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (
        CheckConstraint("settings_id = 1", name="ck_application_settings_singleton"),
        CheckConstraint(
            "global_seed >= 0 AND global_seed <= 4294967295",
            name="ck_application_settings_seed",
        ),
        CheckConstraint(
            "job_polling_interval >= 0.25 AND job_polling_interval <= 60",
            name="ck_application_settings_polling_interval",
        ),
        CheckConstraint(
            "theme_preference IS NULL OR theme_preference IN ('light', 'dark', 'system')",
            name="ck_application_settings_theme_preference",
        ),
    )

###############################################################################
class Dataset(StudioBase):
    """Canonical dataset identity."""

    __tablename__ = "datasets"
    dataset_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
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
class DatasetRecord(StudioBase):
    """Canonical image/report records belonging to a dataset."""

    __tablename__ = "dataset_records"
    record_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
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

###############################################################################
class DatasetVersion(StudioBase):
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
        UniqueConstraint(
            "dataset_id", "version_number", name="uq_dataset_versions_number"
        ),
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
class ProcessingRun(StudioBase):
    """Preprocessing run metadata and configuration."""

    __tablename__ = "processing_runs"
    processing_run_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
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
    tokenizer_revision: Mapped[str | None] = mapped_column(String(64))
    pad_token_id: Mapped[int | None] = mapped_column(Integer)
    start_token_id: Mapped[int | None] = mapped_column(Integer)
    end_token_id: Mapped[int | None] = mapped_column(Integer)
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
class TrainingSample(StudioBase):
    """Processed training samples linked to preprocessing runs and source records."""

    __tablename__ = "training_samples"
    training_sample_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
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
    record: Mapped[DatasetRecord] = relationship(
        "DatasetRecord", back_populates="training_samples"
    )

###############################################################################
class ValidationRun(StudioBase):
    """One immutable validation execution and its configuration.

    Computed values live in the child tables (`validation_run_metrics`,
    `record_text_stats`, `record_image_stats`, `validation_aggregates`,
    `validation_histogram_bins`); `config_json` only holds auxiliary options.
    """

    __tablename__ = "validation_runs"
    validation_run_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_version_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("dataset_versions.dataset_version_id", ondelete="SET NULL"),
        nullable=True,
    )
    executed_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    sample_size: Mapped[float] = mapped_column(Float, nullable=False)
    seed: Mapped[int | None] = mapped_column(BigInteger)
    records_requested: Mapped[int | None] = mapped_column(Integer)
    records_evaluated: Mapped[int | None] = mapped_column(Integer)
    text_vocabulary_size: Mapped[int | None] = mapped_column(Integer)
    tokenizer: Mapped[str | None] = mapped_column(String(255))
    max_report_size: Mapped[int | None] = mapped_column(Integer)
    config_json: Mapped[Any | None] = mapped_column(JSONSequence)
    __table_args__ = (
        Index("ix_validation_runs_dataset_id", "dataset_id"),
        Index("ix_validation_runs_dataset_executed", "dataset_id", "executed_at"),
        CheckConstraint(
            "sample_size > 0 AND sample_size <= 1", name="ck_validation_runs_sample"
        ),
    )
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="validation_runs")

###############################################################################
class ValidationRunMetric(StudioBase):
    """Metric computed by a validation run and the records it covered."""

    __tablename__ = "validation_run_metrics"
    validation_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("validation_runs.validation_run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    metric: Mapped[str] = mapped_column(String(40), primary_key=True)
    records_computed: Mapped[int | None] = mapped_column(Integer)

###############################################################################
class RecordTextStat(StudioBase):
    """Per-record text statistics measured by one validation run."""

    __tablename__ = "record_text_stats"
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
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    unique_words: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    truncated: Mapped[bool | None] = mapped_column(Boolean)
    __table_args__ = (Index("ix_record_text_stats_record_id", "record_id"),)

###############################################################################
class RecordImageStat(StudioBase):
    """Per-record image statistics measured by one validation run."""

    __tablename__ = "record_image_stats"
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
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_value: Mapped[float] = mapped_column(Float, nullable=False)
    median_value: Mapped[float] = mapped_column(Float, nullable=False)
    std_value: Mapped[float] = mapped_column(Float, nullable=False)
    min_value: Mapped[float] = mapped_column(Float, nullable=False)
    max_value: Mapped[float] = mapped_column(Float, nullable=False)
    noise_std: Mapped[float] = mapped_column(Float, nullable=False)
    noise_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    __table_args__ = (Index("ix_record_image_stats_record_id", "record_id"),)

###############################################################################
class ValidationAggregate(StudioBase):
    """Dataset-level distribution summary of one measure in one run."""

    __tablename__ = "validation_aggregates"
    validation_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("validation_runs.validation_run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    domain: Mapped[str] = mapped_column(String(16), primary_key=True)
    measure: Mapped[str] = mapped_column(String(40), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    total: Mapped[float | None] = mapped_column(Float)
    minimum: Mapped[float | None] = mapped_column(Float)
    maximum: Mapped[float | None] = mapped_column(Float)
    mean: Mapped[float | None] = mapped_column(Float)
    median: Mapped[float | None] = mapped_column(Float)
    std: Mapped[float | None] = mapped_column(Float)
    p05: Mapped[float | None] = mapped_column(Float)
    p25: Mapped[float | None] = mapped_column(Float)
    p75: Mapped[float | None] = mapped_column(Float)
    p95: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (
        CheckConstraint(
            "domain IN ('text', 'image', 'token')", name="ck_validation_aggregates_domain"
        ),
        CheckConstraint("count >= 0", name="ck_validation_aggregates_count"),
    )

###############################################################################
class ValidationHistogramBin(StudioBase):
    """One bin of a run-level histogram (pixel intensity, report length)."""

    __tablename__ = "validation_histogram_bins"
    validation_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("validation_runs.validation_run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    histogram: Mapped[str] = mapped_column(String(40), primary_key=True)
    bin_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    bin_lower: Mapped[float] = mapped_column(Float, nullable=False)
    bin_upper: Mapped[float] = mapped_column(Float, nullable=False)
    count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    __table_args__ = (
        CheckConstraint("count >= 0", name="ck_validation_histogram_bins_count"),
    )

###############################################################################
class Checkpoint(CheckpointColumns, StudioBase):
    """Trained checkpoint identity."""

    evaluations: Mapped[list[CheckpointEvaluation]] = relationship(
        "CheckpointEvaluation",
        back_populates="checkpoint",
        passive_deletes=True,
    )

###############################################################################
class CheckpointEvaluation(StudioBase):
    """Latest checkpoint evaluation payload."""

    __tablename__ = "checkpoint_evaluations"
    evaluation_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
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
        Index("ix_checkpoint_evaluations_checkpoint_id", "checkpoint_id"),
    )
    checkpoint: Mapped[Checkpoint] = relationship(
        "Checkpoint", back_populates="evaluations"
    )

###############################################################################
class ModelArchitecturePreset(StudioBase):
    """User-defined model architecture preset."""

    __tablename__ = "model_architecture_presets"
    preset_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    name_key: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    architecture_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (
        UniqueConstraint("name_key", name="uq_model_architecture_presets_name_key"),
    )

###############################################################################
class HiddenArchitecturePreset(StudioBase):
    """Built-in architecture preset removed from the user's preset collection."""

    __tablename__ = "hidden_architecture_presets"
    preset_key: Mapped[str] = mapped_column(String(80), primary_key=True)

###############################################################################
@event.listens_for(Dataset, "before_insert")
def _populate_dataset_name_key(_mapper: Any, _connection: Any, target: Dataset) -> None:
    target.name_key = normalize_key(target.name)

###############################################################################
@event.listens_for(DatasetRecord, "before_insert")
def _populate_image_name_key(
    _mapper: Any, _connection: Any, target: DatasetRecord
) -> None:
    target.image_name_key = normalize_key(target.image_name)


__all__ = [
    "Checkpoint",
    "CheckpointEvaluation",
    "Dataset",
    "DatasetRecord",
    "DatasetVersion",
    "HiddenArchitecturePreset",
    "ModelArchitecturePreset",
    "ProcessingRun",
    "StudioBase",
    "StudioSettingsRecord",
    "TrainingSample",
    "ValidationRun",
]
