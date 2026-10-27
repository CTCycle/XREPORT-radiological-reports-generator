"""Create the XREPORT Studio schema and seed its settings row."""

from datetime import datetime, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import server.repositories.schemas.types

revision: str = "9d4d3bce4ede"
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
        sa.Column("theme_preference", sa.String(length=8), nullable=True),
        sa.Column(
            "updated_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
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
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("checkpoint_id"),
        sa.UniqueConstraint("name_key", name="uq_checkpoints_name_key"),
    )
    op.create_table(
        "datasets",
        sa.Column("dataset_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("name_key", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("dataset_id"),
        sa.UniqueConstraint("name_key", name="uq_datasets_name_key"),
    )
    op.create_table(
        "checkpoint_evaluations",
        sa.Column("evaluation_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("checkpoint_id", sa.Integer(), nullable=False),
        sa.Column(
            "executed_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "metrics_json",
            server.repositories.schemas.types.JSONSequence(),
            nullable=False,
        ),
        sa.Column(
            "metric_configs_json",
            server.repositories.schemas.types.JSONSequence(),
            nullable=False,
        ),
        sa.Column(
            "results_json",
            server.repositories.schemas.types.JSONSequence(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["checkpoint_id"], ["checkpoints.checkpoint_id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("evaluation_id"),
    )
    with op.batch_alter_table("checkpoint_evaluations", schema=None) as batch_op:
        batch_op.create_index(
            "ix_checkpoint_evaluations_checkpoint_id", ["checkpoint_id"], unique=False
        )

    op.create_table(
        "dataset_versions",
        sa.Column(
            "dataset_version_id", sa.Integer(), autoincrement=True, nullable=False
        ),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("record_count", sa.Integer(), nullable=False),
        sa.Column(
            "imported_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "record_count >= 0", name="ck_dataset_versions_record_count"
        ),
        sa.CheckConstraint("version_number > 0", name="ck_dataset_versions_number"),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.dataset_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("dataset_version_id"),
        sa.UniqueConstraint(
            "dataset_id", "content_hash", name="uq_dataset_versions_hash"
        ),
        sa.UniqueConstraint(
            "dataset_id", "version_number", name="uq_dataset_versions_number"
        ),
    )
    with op.batch_alter_table("dataset_versions", schema=None) as batch_op:
        batch_op.create_index(
            "ix_dataset_versions_dataset_latest",
            ["dataset_id", "version_number"],
            unique=False,
        )

    op.create_table(
        "processing_runs",
        sa.Column(
            "processing_run_id", sa.Integer(), autoincrement=True, nullable=False
        ),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("source_dataset_id", sa.Integer(), nullable=True),
        sa.Column("config_hash", sa.String(), nullable=False),
        sa.Column(
            "executed_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("sample_size", sa.Float(), nullable=False),
        sa.Column("validation_size", sa.Float(), nullable=False),
        sa.Column("split_seed", sa.Integer(), nullable=False),
        sa.Column("vocabulary_size", sa.Integer(), nullable=True),
        sa.Column("max_report_size", sa.Integer(), nullable=False),
        sa.Column("tokenizer", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.dataset_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_dataset_id"], ["datasets.dataset_id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("processing_run_id"),
    )
    with op.batch_alter_table("processing_runs", schema=None) as batch_op:
        batch_op.create_index(
            "ix_processing_runs_config_hash", ["config_hash"], unique=False
        )
        batch_op.create_index(
            "ix_processing_runs_dataset_id", ["dataset_id"], unique=False
        )

    op.create_table(
        "validation_runs",
        sa.Column(
            "validation_run_id", sa.Integer(), autoincrement=True, nullable=False
        ),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column(
            "executed_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("sample_size", sa.Float(), nullable=False),
        sa.Column(
            "metrics_json",
            server.repositories.schemas.types.JSONSequence(),
            nullable=False,
        ),
        sa.Column(
            "artifacts_json",
            server.repositories.schemas.types.JSONSequence(),
            nullable=True,
        ),
        sa.Column("text_count", sa.Integer(), nullable=True),
        sa.Column("text_total_words", sa.Integer(), nullable=True),
        sa.Column("text_unique_words", sa.Integer(), nullable=True),
        sa.Column("text_avg_words", sa.Float(), nullable=True),
        sa.Column("text_min_words", sa.Integer(), nullable=True),
        sa.Column("text_max_words", sa.Integer(), nullable=True),
        sa.Column("image_count", sa.Integer(), nullable=True),
        sa.Column("image_mean_height", sa.Float(), nullable=True),
        sa.Column("image_mean_width", sa.Float(), nullable=True),
        sa.Column("image_mean_value", sa.Float(), nullable=True),
        sa.Column("image_std_value", sa.Float(), nullable=True),
        sa.Column("image_mean_noise_std", sa.Float(), nullable=True),
        sa.Column("image_mean_noise_ratio", sa.Float(), nullable=True),
        sa.Column(
            "pixel_bins_json",
            server.repositories.schemas.types.JSONSequence(),
            nullable=True,
        ),
        sa.Column(
            "pixel_counts_json",
            server.repositories.schemas.types.JSONSequence(),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.dataset_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("validation_run_id"),
    )
    with op.batch_alter_table("validation_runs", schema=None) as batch_op:
        batch_op.create_index(
            "ix_validation_runs_dataset_executed",
            ["dataset_id", "executed_at"],
            unique=False,
        )
        batch_op.create_index(
            "ix_validation_runs_dataset_id", ["dataset_id"], unique=False
        )

    op.create_table(
        "dataset_records",
        sa.Column("record_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("dataset_version_id", sa.Integer(), nullable=True),
        sa.Column("image_name", sa.String(length=512), nullable=False),
        sa.Column("image_name_key", sa.String(length=512), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("report_text", sa.Text(), nullable=False),
        sa.Column("row_order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.dataset_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.dataset_version_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("record_id"),
        sa.UniqueConstraint(
            "dataset_version_id",
            "image_name_key",
            name="uq_dataset_records_dataset_image_key",
        ),
        sa.UniqueConstraint(
            "dataset_version_id", "row_order", name="uq_dataset_records_dataset_order"
        ),
    )
    with op.batch_alter_table("dataset_records", schema=None) as batch_op:
        batch_op.create_index(
            "ix_dataset_records_dataset_image",
            ["dataset_id", "image_name"],
            unique=False,
        )
        batch_op.create_index(
            "ix_dataset_records_dataset_order",
            ["dataset_id", "row_order"],
            unique=False,
        )

    op.create_table(
        "training_samples",
        sa.Column(
            "training_sample_id", sa.Integer(), autoincrement=True, nullable=False
        ),
        sa.Column("processing_run_id", sa.Integer(), nullable=False),
        sa.Column("record_id", sa.Integer(), nullable=False),
        sa.Column("split", sa.String(), nullable=False),
        sa.Column(
            "tokens_json",
            server.repositories.schemas.types.JSONSequence(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "split IN ('train', 'validation')", name="ck_training_samples_split"
        ),
        sa.ForeignKeyConstraint(
            ["processing_run_id"],
            ["processing_runs.processing_run_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["record_id"], ["dataset_records.record_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("training_sample_id"),
        sa.UniqueConstraint(
            "processing_run_id", "record_id", name="uq_training_samples_run_record"
        ),
    )
    with op.batch_alter_table("training_samples", schema=None) as batch_op:
        batch_op.create_index(
            "ix_training_samples_run_split",
            ["processing_run_id", "split"],
            unique=False,
        )

###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("training_samples", schema=None) as batch_op:
        batch_op.drop_index("ix_training_samples_run_split")

    op.drop_table("training_samples")
    with op.batch_alter_table("dataset_records", schema=None) as batch_op:
        batch_op.drop_index("ix_dataset_records_dataset_order")
        batch_op.drop_index("ix_dataset_records_dataset_image")

    op.drop_table("dataset_records")
    with op.batch_alter_table("validation_runs", schema=None) as batch_op:
        batch_op.drop_index("ix_validation_runs_dataset_id")
        batch_op.drop_index("ix_validation_runs_dataset_executed")

    op.drop_table("validation_runs")
    with op.batch_alter_table("processing_runs", schema=None) as batch_op:
        batch_op.drop_index("ix_processing_runs_dataset_id")
        batch_op.drop_index("ix_processing_runs_config_hash")

    op.drop_table("processing_runs")
    with op.batch_alter_table("dataset_versions", schema=None) as batch_op:
        batch_op.drop_index("ix_dataset_versions_dataset_latest")

    op.drop_table("dataset_versions")
    with op.batch_alter_table("checkpoint_evaluations", schema=None) as batch_op:
        batch_op.drop_index("ix_checkpoint_evaluations_checkpoint_id")

    op.drop_table("checkpoint_evaluations")
    op.drop_table("datasets")
    op.drop_table("checkpoints")
    op.drop_table("application_settings")
