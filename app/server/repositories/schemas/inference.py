"""XREPORT Inference ORM schema: settings, imported checkpoints, and report history."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from server.repositories.schemas.checkpoint import CheckpointColumns
from server.repositories.schemas.types import JSONSequence, UTCDateTime

###############################################################################
class InferenceBase(DeclarativeBase):
    pass

###############################################################################
class InferenceSettingsRecord(InferenceBase):
    """Singleton persisted XREPORT Inference settings row."""

    __tablename__ = "application_settings"
    settings_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    global_seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    job_polling_interval: Mapped[float] = mapped_column(Float, nullable=False)
    inference_model_timeout: Mapped[int] = mapped_column(BigInteger, nullable=False)
    inference_model_idle_timeout: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=300
    )
    inference_max_new_tokens_deterministic: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0
    )
    inference_max_new_tokens_concise: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0
    )
    inference_max_new_tokens_detailed: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0
    )
    inference_detailed_num_beams: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0
    )
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
            "inference_model_timeout >= 1",
            name="ck_application_settings_model_timeout",
        ),
        CheckConstraint(
            "inference_model_idle_timeout >= 0",
            name="ck_application_settings_model_idle_timeout",
        ),
        CheckConstraint(
            "inference_max_new_tokens_deterministic >= 0 AND inference_max_new_tokens_deterministic <= 4096"
            " AND inference_max_new_tokens_concise >= 0 AND inference_max_new_tokens_concise <= 4096"
            " AND inference_max_new_tokens_detailed >= 0 AND inference_max_new_tokens_detailed <= 4096",
            name="ck_application_settings_max_new_tokens",
        ),
        CheckConstraint(
            "inference_detailed_num_beams >= 0 AND inference_detailed_num_beams <= 8",
            name="ck_application_settings_detailed_num_beams",
        ),
        CheckConstraint(
            "theme_preference IS NULL OR theme_preference IN ('light', 'dark', 'system')",
            name="ck_application_settings_theme_preference",
        ),
    )

###############################################################################
class Checkpoint(CheckpointColumns, InferenceBase):
    """Imported checkpoint identity."""

    inference_runs: Mapped[list[InferenceRun]] = relationship(
        "InferenceRun",
        back_populates="checkpoint",
        passive_deletes=True,
    )

###############################################################################
class InferenceRun(InferenceBase):
    """Inference execution metadata."""

    __tablename__ = "inference_runs"
    inference_run_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    checkpoint_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("checkpoints.checkpoint_id", ondelete="RESTRICT"),
        nullable=True,
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    model_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    generation_profile: Mapped[str] = mapped_column(String(32), nullable=False)
    generation_config_json: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    clinical_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="succeeded")
    execution_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
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
    checkpoint: Mapped[Checkpoint | None] = relationship(
        "Checkpoint", back_populates="inference_runs"
    )
    reports: Mapped[list[InferenceReport]] = relationship(
        "InferenceReport",
        back_populates="inference_run",
        cascade="all, delete-orphan",
    )

###############################################################################
class InferenceReport(InferenceBase):
    """Generated reports linked to inference runs."""

    __tablename__ = "inference_reports"
    inference_report_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    inference_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("inference_runs.inference_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    input_image_name: Mapped[str] = mapped_column(String, nullable=False)
    input_image_name_key: Mapped[str] = mapped_column(String(512), nullable=False)
    image_index: Mapped[int] = mapped_column(Integer, nullable=False)
    generated_report: Mapped[str] = mapped_column(Text, nullable=False)
    edited_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    edited_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
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
    inference_run: Mapped[InferenceRun] = relationship(
        "InferenceRun", back_populates="reports"
    )


__all__ = [
    "Checkpoint",
    "InferenceBase",
    "InferenceReport",
    "InferenceRun",
    "InferenceSettingsRecord",
]
