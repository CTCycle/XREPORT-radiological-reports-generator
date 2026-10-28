"""Create the XREPORT Inference schema and seed its settings row."""

from datetime import datetime, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import xreport_shared.repositories.schemas.types

revision: str = "36d142391099"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


###############################################################################
def upgrade() -> None:
    op.create_table(
        "application_settings",
        sa.Column("settings_id", sa.Integer(), nullable=False),
        sa.Column("global_seed", sa.BigInteger(), nullable=False),
        sa.Column("job_polling_interval", sa.Float(), nullable=False),
        sa.Column("inference_model_timeout", sa.BigInteger(), nullable=False),
        sa.Column("inference_model_idle_timeout", sa.BigInteger(), nullable=False),
        sa.Column(
            "inference_max_new_tokens_deterministic", sa.BigInteger(), nullable=False
        ),
        sa.Column("inference_max_new_tokens_concise", sa.BigInteger(), nullable=False),
        sa.Column("inference_max_new_tokens_detailed", sa.BigInteger(), nullable=False),
        sa.Column("inference_detailed_num_beams", sa.BigInteger(), nullable=False),
        sa.Column("theme_preference", sa.String(length=8), nullable=True),
        sa.Column(
            "updated_at",
            xreport_shared.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "theme_preference IS NULL OR theme_preference IN ('light', 'dark', 'system')",
            name="ck_application_settings_theme_preference",
        ),
        sa.CheckConstraint(
            "global_seed >= 0 AND global_seed <= 4294967295",
            name="ck_application_settings_seed",
        ),
        sa.CheckConstraint(
            "inference_detailed_num_beams >= 0 AND inference_detailed_num_beams <= 8",
            name="ck_application_settings_detailed_num_beams",
        ),
        sa.CheckConstraint(
            "inference_max_new_tokens_deterministic >= 0 AND inference_max_new_tokens_deterministic <= 4096 AND inference_max_new_tokens_concise >= 0 AND inference_max_new_tokens_concise <= 4096 AND inference_max_new_tokens_detailed >= 0 AND inference_max_new_tokens_detailed <= 4096",
            name="ck_application_settings_max_new_tokens",
        ),
        sa.CheckConstraint(
            "inference_model_idle_timeout >= 0",
            name="ck_application_settings_model_idle_timeout",
        ),
        sa.CheckConstraint(
            "inference_model_timeout >= 1", name="ck_application_settings_model_timeout"
        ),
        sa.CheckConstraint(
            "job_polling_interval >= 0.25 AND job_polling_interval <= 60",
            name="ck_application_settings_polling_interval",
        ),
        sa.CheckConstraint("settings_id = 1", name="ck_application_settings_singleton"),
        sa.PrimaryKeyConstraint("settings_id"),
    )
    application_settings = sa.table(
        "application_settings",
        sa.column("settings_id", sa.Integer()),
        sa.column("global_seed", sa.BigInteger()),
        sa.column("job_polling_interval", sa.Float()),
        sa.column("inference_model_timeout", sa.BigInteger()),
        sa.column("inference_model_idle_timeout", sa.BigInteger()),
        sa.column("inference_max_new_tokens_deterministic", sa.BigInteger()),
        sa.column("inference_max_new_tokens_concise", sa.BigInteger()),
        sa.column("inference_max_new_tokens_detailed", sa.BigInteger()),
        sa.column("inference_detailed_num_beams", sa.BigInteger()),
        sa.column("theme_preference", sa.String(length=8)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    op.bulk_insert(
        application_settings,
        [
            {
                "settings_id": 1,
                "global_seed": 42,
                "job_polling_interval": 1.0,
                "inference_model_timeout": 600,
                "inference_model_idle_timeout": 300,
                "inference_max_new_tokens_deterministic": 0,
                "inference_max_new_tokens_concise": 0,
                "inference_max_new_tokens_detailed": 0,
                "inference_detailed_num_beams": 0,
                "theme_preference": None,
                "updated_at": datetime.now(timezone.utc),
            }
        ],
    )
    op.create_table(
        "checkpoints",
        sa.Column("checkpoint_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("name_key", sa.String(length=255), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            xreport_shared.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            xreport_shared.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("checkpoint_id"),
        sa.UniqueConstraint("name_key", name="uq_checkpoints_name_key"),
    )
    op.create_table(
        "inference_runs",
        sa.Column("inference_run_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("checkpoint_id", sa.Integer(), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model_ref", sa.String(length=512), nullable=False),
        sa.Column("model_revision", sa.String(length=64), nullable=True),
        sa.Column("generation_profile", sa.String(length=32), nullable=False),
        sa.Column(
            "generation_config_json",
            xreport_shared.repositories.schemas.types.JSONSequence(),
            nullable=False,
        ),
        sa.Column("clinical_context", sa.Text(), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("execution_time_seconds", sa.Float(), nullable=True),
        sa.Column(
            "executed_at",
            xreport_shared.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_inference_runs_status",
        ),
        sa.ForeignKeyConstraint(
            ["checkpoint_id"], ["checkpoints.checkpoint_id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("inference_run_id"),
        sa.UniqueConstraint("request_id", name="uq_inference_runs_request_id"),
    )
    op.create_table(
        "inference_reports",
        sa.Column(
            "inference_report_id", sa.Integer(), autoincrement=True, nullable=False
        ),
        sa.Column("inference_run_id", sa.Integer(), nullable=False),
        sa.Column("input_image_name", sa.String(), nullable=False),
        sa.Column("input_image_name_key", sa.String(length=512), nullable=False),
        sa.Column("image_index", sa.Integer(), nullable=False),
        sa.Column("generated_report", sa.Text(), nullable=False),
        sa.Column("edited_report", sa.Text(), nullable=True),
        sa.Column(
            "edited_at",
            xreport_shared.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint("image_index >= 0", name="ck_inference_reports_image_index"),
        sa.ForeignKeyConstraint(
            ["inference_run_id"],
            ["inference_runs.inference_run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("inference_report_id"),
        sa.UniqueConstraint(
            "inference_run_id", "image_index", name="uq_inference_reports_run_index"
        ),
        sa.UniqueConstraint(
            "inference_run_id",
            "input_image_name_key",
            name="uq_inference_reports_run_image",
        ),
    )


###############################################################################
def downgrade() -> None:
    op.drop_table("inference_reports")
    op.drop_table("inference_runs")
    op.drop_table("checkpoints")
    op.drop_table("application_settings")
