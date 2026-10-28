"""Store validation statistics in relational tables instead of run-level JSON."""

from typing import Any, Sequence, Union

from alembic import op
import sqlalchemy as sa

import xreport_shared.repositories.schemas.types

revision: str = "c7e2a94f1b08"
down_revision: Union[str, None] = "b3c1d5a7e9f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

LEGACY_COLUMNS = (
    ("metrics_json", sa.JSON()),
    ("artifacts_json", sa.JSON()),
    ("text_count", sa.Integer()),
    ("text_total_words", sa.Integer()),
    ("text_unique_words", sa.Integer()),
    ("text_avg_words", sa.Float()),
    ("text_min_words", sa.Integer()),
    ("text_max_words", sa.Integer()),
    ("image_count", sa.Integer()),
    ("image_mean_height", sa.Float()),
    ("image_mean_width", sa.Float()),
    ("image_mean_value", sa.Float()),
    ("image_std_value", sa.Float()),
    ("image_mean_noise_std", sa.Float()),
    ("image_mean_noise_ratio", sa.Float()),
    ("pixel_bins_json", sa.JSON()),
    ("pixel_counts_json", sa.JSON()),
)


###############################################################################
def _run_fk(name: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["validation_run_id"],
        ["validation_runs.validation_run_id"],
        name=name,
        ondelete="CASCADE",
    )


###############################################################################
def _record_fk(name: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["record_id"],
        ["dataset_records.record_id"],
        name=name,
        ondelete="CASCADE",
    )


###############################################################################
def _legacy_table() -> sa.Table:
    return sa.table(
        "validation_runs",
        sa.column("validation_run_id", sa.Integer()),
        *[sa.column(name, kind) for name, kind in LEGACY_COLUMNS],
    )


###############################################################################
def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


###############################################################################
def _scalar_measure(
    run_id: int, domain: str, measure: str, count: int, **values: Any
) -> dict[str, Any]:
    return {
        "validation_run_id": run_id,
        "domain": domain,
        "measure": measure,
        "count": count,
        "total": values.get("total"),
        "minimum": values.get("minimum"),
        "maximum": values.get("maximum"),
        "mean": values.get("mean"),
        "median": None,
        "std": None,
        "p05": None,
        "p25": None,
        "p75": None,
        "p95": None,
    }


###############################################################################
def _backfill(bind: sa.engine.Connection) -> None:
    """Copy legacy aggregate values into the relational tables.

    Legacy runs have no per-record rows, so distribution columns that cannot be
    derived (median, std, percentiles) stay NULL instead of being invented.
    """
    legacy = _legacy_table()
    metrics_table = sa.table(
        "validation_run_metrics",
        sa.column("validation_run_id"),
        sa.column("metric"),
        sa.column("records_computed"),
    )
    aggregates_table = sa.table(
        "validation_aggregates",
        *[
            sa.column(name)
            for name in (
                "validation_run_id",
                "domain",
                "measure",
                "count",
                "total",
                "minimum",
                "maximum",
                "mean",
                "median",
                "std",
                "p05",
                "p25",
                "p75",
                "p95",
            )
        ],
    )
    bins_table = sa.table(
        "validation_histogram_bins",
        *[
            sa.column(name)
            for name in (
                "validation_run_id",
                "histogram",
                "bin_index",
                "bin_lower",
                "bin_upper",
                "count",
            )
        ],
    )
    runs_table = sa.table(
        "validation_runs",
        sa.column("validation_run_id"),
        sa.column("records_evaluated"),
        sa.column("text_vocabulary_size"),
    )
    metric_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for row in bind.execute(sa.select(legacy)).mappings():
        run_id = int(row["validation_run_id"])
        metrics = [str(item) for item in _as_list(row["metrics_json"])]
        evaluated = 0
        vocabulary_size = None
        if "text_statistics" in metrics:
            text_count = int(row["text_count"] or 0)
            metric_rows.append(
                {
                    "validation_run_id": run_id,
                    "metric": "text_statistics",
                    "records_computed": text_count,
                }
            )
            if text_count > 0:
                evaluated = max(evaluated, text_count)
                vocabulary_size = row["text_unique_words"]
                aggregate_rows.append(
                    _scalar_measure(
                        run_id,
                        "text",
                        "words_per_report",
                        text_count,
                        total=row["text_total_words"],
                        minimum=row["text_min_words"],
                        maximum=row["text_max_words"],
                        mean=row["text_avg_words"],
                    )
                )
        if "image_statistics" in metrics:
            image_count = int(row["image_count"] or 0)
            metric_rows.append(
                {
                    "validation_run_id": run_id,
                    "metric": "image_statistics",
                    "records_computed": image_count,
                }
            )
            if image_count > 0:
                evaluated = max(evaluated, image_count)
                for measure, column in (
                    ("height", "image_mean_height"),
                    ("width", "image_mean_width"),
                    ("mean_intensity", "image_mean_value"),
                    ("std_intensity", "image_std_value"),
                    ("noise_std", "image_mean_noise_std"),
                    ("noise_ratio", "image_mean_noise_ratio"),
                ):
                    aggregate_rows.append(
                        _scalar_measure(
                            run_id, "image", measure, image_count, mean=row[column]
                        )
                    )
        if "pixels_distribution" in metrics:
            metric_rows.append(
                {
                    "validation_run_id": run_id,
                    "metric": "pixels_distribution",
                    "records_computed": None,
                }
            )
            lowers = _as_list(row["pixel_bins_json"])
            counts = _as_list(row["pixel_counts_json"])
            for index, (lower, count) in enumerate(zip(lowers, counts)):
                bin_rows.append(
                    {
                        "validation_run_id": run_id,
                        "histogram": "pixel_intensity",
                        "bin_index": index,
                        "bin_lower": float(lower),
                        "bin_upper": float(lower) + 1.0,
                        "count": int(count),
                    }
                )
        bind.execute(
            sa.update(runs_table)
            .where(runs_table.c.validation_run_id == run_id)
            .values(
                records_evaluated=evaluated or None,
                text_vocabulary_size=vocabulary_size,
            )
        )
    for table, rows in (
        (metrics_table, metric_rows),
        (aggregates_table, aggregate_rows),
        (bins_table, bin_rows),
    ):
        if rows:
            bind.execute(sa.insert(table), rows)


###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("validation_runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("dataset_version_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("seed", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("records_requested", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("records_evaluated", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("text_vocabulary_size", sa.Integer(), nullable=True)
        )
        batch_op.add_column(sa.Column("tokenizer", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("max_report_size", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "config_json", xreport_shared.repositories.schemas.types.JSONSequence(), nullable=True
            )
        )
        batch_op.create_foreign_key(
            "fk_validation_runs_dataset_version_id",
            "dataset_versions",
            ["dataset_version_id"],
            ["dataset_version_id"],
            ondelete="SET NULL",
        )
        batch_op.create_check_constraint(
            "ck_validation_runs_sample", "sample_size > 0 AND sample_size <= 1"
        )

    op.create_table(
        "validation_run_metrics",
        sa.Column("validation_run_id", sa.Integer(), nullable=False),
        sa.Column("metric", sa.String(length=40), nullable=False),
        sa.Column("records_computed", sa.Integer(), nullable=True),
        _run_fk("fk_validation_run_metrics_run"),
        sa.PrimaryKeyConstraint("validation_run_id", "metric"),
    )
    op.create_table(
        "record_text_stats",
        sa.Column("validation_run_id", sa.Integer(), nullable=False),
        sa.Column("record_id", sa.Integer(), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("unique_words", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("truncated", sa.Boolean(), nullable=True),
        _run_fk("fk_record_text_stats_run"),
        _record_fk("fk_record_text_stats_record"),
        sa.PrimaryKeyConstraint("validation_run_id", "record_id"),
    )
    with op.batch_alter_table("record_text_stats", schema=None) as batch_op:
        batch_op.create_index("ix_record_text_stats_record_id", ["record_id"])
    op.create_table(
        "record_image_stats",
        sa.Column("validation_run_id", sa.Integer(), nullable=False),
        sa.Column("record_id", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("mean_value", sa.Float(), nullable=False),
        sa.Column("median_value", sa.Float(), nullable=False),
        sa.Column("std_value", sa.Float(), nullable=False),
        sa.Column("min_value", sa.Float(), nullable=False),
        sa.Column("max_value", sa.Float(), nullable=False),
        sa.Column("noise_std", sa.Float(), nullable=False),
        sa.Column("noise_ratio", sa.Float(), nullable=False),
        _run_fk("fk_record_image_stats_run"),
        _record_fk("fk_record_image_stats_record"),
        sa.PrimaryKeyConstraint("validation_run_id", "record_id"),
    )
    with op.batch_alter_table("record_image_stats", schema=None) as batch_op:
        batch_op.create_index("ix_record_image_stats_record_id", ["record_id"])
    op.create_table(
        "validation_aggregates",
        sa.Column("validation_run_id", sa.Integer(), nullable=False),
        sa.Column("domain", sa.String(length=16), nullable=False),
        sa.Column("measure", sa.String(length=40), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("total", sa.Float(), nullable=True),
        sa.Column("minimum", sa.Float(), nullable=True),
        sa.Column("maximum", sa.Float(), nullable=True),
        sa.Column("mean", sa.Float(), nullable=True),
        sa.Column("median", sa.Float(), nullable=True),
        sa.Column("std", sa.Float(), nullable=True),
        sa.Column("p05", sa.Float(), nullable=True),
        sa.Column("p25", sa.Float(), nullable=True),
        sa.Column("p75", sa.Float(), nullable=True),
        sa.Column("p95", sa.Float(), nullable=True),
        sa.CheckConstraint(
            "domain IN ('text', 'image', 'token')",
            name="ck_validation_aggregates_domain",
        ),
        sa.CheckConstraint("count >= 0", name="ck_validation_aggregates_count"),
        _run_fk("fk_validation_aggregates_run"),
        sa.PrimaryKeyConstraint("validation_run_id", "domain", "measure"),
    )
    op.create_table(
        "validation_histogram_bins",
        sa.Column("validation_run_id", sa.Integer(), nullable=False),
        sa.Column("histogram", sa.String(length=40), nullable=False),
        sa.Column("bin_index", sa.Integer(), nullable=False),
        sa.Column("bin_lower", sa.Float(), nullable=False),
        sa.Column("bin_upper", sa.Float(), nullable=False),
        sa.Column("count", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("count >= 0", name="ck_validation_histogram_bins_count"),
        _run_fk("fk_validation_histogram_bins_run"),
        sa.PrimaryKeyConstraint("validation_run_id", "histogram", "bin_index"),
    )

    _backfill(op.get_bind())

    with op.batch_alter_table("validation_runs", schema=None) as batch_op:
        for name, _kind in LEGACY_COLUMNS:
            batch_op.drop_column(name)


###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("validation_runs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "metrics_json",
                xreport_shared.repositories.schemas.types.JSONSequence(),
                nullable=False,
                server_default="[]",
            )
        )
        for name, kind in LEGACY_COLUMNS[1:]:
            column_type = (
                xreport_shared.repositories.schemas.types.JSONSequence()
                if isinstance(kind, sa.JSON)
                else kind
            )
            batch_op.add_column(sa.Column(name, column_type, nullable=True))

    bind = op.get_bind()
    runs = _legacy_table()
    run_ids = [
        int(item)
        for item in bind.execute(sa.text("SELECT validation_run_id FROM validation_runs"))
        .scalars()
        .all()
    ]
    for run_id in run_ids:
        metrics = (
            bind.execute(
                sa.text(
                    "SELECT metric FROM validation_run_metrics "
                    "WHERE validation_run_id = :run ORDER BY metric"
                ),
                {"run": run_id},
            )
            .scalars()
            .all()
        )
        aggregates = {
            (item["domain"], item["measure"]): item
            for item in bind.execute(
                sa.text(
                    "SELECT domain, measure, count, total, minimum, maximum, mean "
                    "FROM validation_aggregates WHERE validation_run_id = :run"
                ),
                {"run": run_id},
            ).mappings()
        }
        bins = bind.execute(
            sa.text(
                "SELECT bin_lower, count FROM validation_histogram_bins "
                "WHERE validation_run_id = :run AND histogram = 'pixel_intensity' "
                "ORDER BY bin_index"
            ),
            {"run": run_id},
        ).all()
        values: dict[str, Any] = {
            "metrics_json": list(metrics),
            "artifacts_json": {},
            "pixel_bins_json": [int(item[0]) for item in bins] or None,
            "pixel_counts_json": [int(item[1]) for item in bins] or None,
        }
        text = aggregates.get(("text", "words_per_report"))
        if text is not None:
            values.update(
                text_count=text["count"],
                text_total_words=int(text["total"] or 0),
                text_min_words=int(text["minimum"] or 0),
                text_max_words=int(text["maximum"] or 0),
                text_avg_words=text["mean"],
            )
        image_count = 0
        for key, column in (
            ("height", "image_mean_height"),
            ("width", "image_mean_width"),
            ("mean_intensity", "image_mean_value"),
            ("std_intensity", "image_std_value"),
            ("noise_std", "image_mean_noise_std"),
            ("noise_ratio", "image_mean_noise_ratio"),
        ):
            item = aggregates.get(("image", key))
            if item is not None:
                image_count = int(item["count"])
                values[column] = item["mean"]
        if image_count:
            values["image_count"] = image_count
        bind.execute(
            sa.update(runs)
            .where(runs.c.validation_run_id == run_id)
            .values(**values)
        )

    op.drop_table("validation_histogram_bins")
    op.drop_table("validation_aggregates")
    with op.batch_alter_table("record_image_stats", schema=None) as batch_op:
        batch_op.drop_index("ix_record_image_stats_record_id")
    op.drop_table("record_image_stats")
    with op.batch_alter_table("record_text_stats", schema=None) as batch_op:
        batch_op.drop_index("ix_record_text_stats_record_id")
    op.drop_table("record_text_stats")
    op.drop_table("validation_run_metrics")
    with op.batch_alter_table("validation_runs", schema=None) as batch_op:
        batch_op.drop_constraint("ck_validation_runs_sample", type_="check")
        batch_op.drop_constraint(
            "fk_validation_runs_dataset_version_id", type_="foreignkey"
        )
        for name in (
            "config_json",
            "max_report_size",
            "tokenizer",
            "text_vocabulary_size",
            "records_evaluated",
            "records_requested",
            "seed",
            "dataset_version_id",
        ):
            batch_op.drop_column(name)
