from __future__ import annotations

import json
from pathlib import Path

import sqlalchemy
from alembic import command
from alembic.config import Config

from xreport_shared.repositories.database.schema import ALEMBIC_INI, application_schema

LEGACY_REVISION = "b3c1d5a7e9f2"
STUDIO = application_schema("studio")

###############################################################################
def _config(path: Path) -> Config:
    config = Config(
        str(ALEMBIC_INI),
        ini_section=STUDIO.migrations_dir.name,
    )
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path.as_posix()}")
    return config

###############################################################################
def _seed_legacy_runs(engine: sqlalchemy.Engine) -> None:
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO datasets (dataset_id, name, name_key, created_at) "
            "VALUES (1, 'demo', 'demo', '2026-01-01 00:00:00')"
        )
        statement = sqlalchemy.text(
            "INSERT INTO validation_runs (dataset_id, executed_at, sample_size, "
            "metrics_json, artifacts_json, text_count, text_total_words, "
            "text_unique_words, text_avg_words, text_min_words, text_max_words, "
            "image_count, image_mean_height, image_mean_width, image_mean_value, "
            "image_std_value, image_mean_noise_std, image_mean_noise_ratio, "
            "pixel_bins_json, pixel_counts_json) VALUES (1, :at, :size, :metrics, "
            "'{}', :tc, :tw, :tu, :ta, :tmin, :tmax, :ic, :ih, :iw, :im, :istd, "
            ":ins, :inr, :bins, :counts)"
        )
        full = {
            "at": "2026-02-01 10:00:00",
            "size": 0.5,
            "metrics": json.dumps(
                ["text_statistics", "image_statistics", "pixels_distribution"]
            ),
            "tc": 6, "tw": 60, "tu": 25, "ta": 10.0, "tmin": 4, "tmax": 20,
            "ic": 6, "ih": 512.0, "iw": 480.0, "im": 100.5, "istd": 40.25,
            "ins": 2.5, "inr": 0.06,
            "bins": json.dumps(list(range(256))),
            "counts": json.dumps([1] * 256),
        }
        text_only = {
            **full,
            "at": "2026-03-01 10:00:00",
            "metrics": json.dumps(["text_statistics"]),
            "ic": 0, "ih": 0.0, "iw": 0.0, "im": 0.0, "istd": 0.0,
            "ins": 0.0, "inr": 0.0,
            "bins": json.dumps([]), "counts": json.dumps([]),
        }
        connection.execute(statement, [full, text_only])

###############################################################################
def test_legacy_statistics_are_backfilled_into_relational_tables(tmp_path) -> None:
    path = tmp_path / "database.db"
    command.upgrade(_config(path), LEGACY_REVISION)
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    try:
        _seed_legacy_runs(engine)
        command.upgrade(_config(path), "heads")
        with engine.connect() as connection:
            columns = {
                item["name"] for item in sqlalchemy.inspect(connection).get_columns("validation_runs")
            }
            assert "metrics_json" not in columns and "pixel_bins_json" not in columns
            runs = connection.exec_driver_sql(
                "SELECT validation_run_id, records_evaluated, text_vocabulary_size "
                "FROM validation_runs ORDER BY validation_run_id"
            ).all()
            assert runs == [(1, 6, 25), (2, 6, 25)]
            metrics = connection.exec_driver_sql(
                "SELECT validation_run_id, metric, records_computed "
                "FROM validation_run_metrics ORDER BY 1, 2"
            ).all()
            assert metrics == [
                (1, "image_statistics", 6),
                (1, "pixels_distribution", None),
                (1, "text_statistics", 6),
                (2, "text_statistics", 6),
            ]
            text = connection.exec_driver_sql(
                "SELECT count, total, minimum, maximum, mean, median FROM "
                "validation_aggregates WHERE validation_run_id = 1 AND domain = 'text'"
            ).one()
            assert text == (6, 60.0, 4.0, 20.0, 10.0, None)
            # The text-only run never measured images, so no zero rows are invented.
            assert connection.exec_driver_sql(
                "SELECT COUNT(*) FROM validation_aggregates "
                "WHERE validation_run_id = 2 AND domain = 'image'"
            ).scalar_one() == 0
            assert connection.exec_driver_sql(
                "SELECT mean FROM validation_aggregates WHERE validation_run_id = 1 "
                "AND measure = 'mean_intensity'"
            ).scalar_one() == 100.5
            assert connection.exec_driver_sql(
                "SELECT COUNT(*), SUM(count) FROM validation_histogram_bins "
                "WHERE validation_run_id = 1 AND histogram = 'pixel_intensity'"
            ).one() == (256, 256)
    finally:
        engine.dispose()

###############################################################################
def test_downgrade_restores_legacy_columns_and_values(tmp_path) -> None:
    path = tmp_path / "database.db"
    command.upgrade(_config(path), LEGACY_REVISION)
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    try:
        _seed_legacy_runs(engine)
        command.upgrade(_config(path), "heads")
        command.downgrade(_config(path), LEGACY_REVISION)
        with engine.connect() as connection:
            row = connection.exec_driver_sql(
                "SELECT metrics_json, text_total_words, image_count, image_mean_value "
                "FROM validation_runs WHERE validation_run_id = 1"
            ).one()
            assert json.loads(row[0]) == [
                "image_statistics",
                "pixels_distribution",
                "text_statistics",
            ]
            assert row[1:] == (60, 6, 100.5)
            assert "validation_aggregates" not in sqlalchemy.inspect(connection).get_table_names()
    finally:
        engine.dispose()
