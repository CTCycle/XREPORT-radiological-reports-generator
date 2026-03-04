from __future__ import annotations

import re
from typing import Any

import pandas as pd

from XREPORT.server.common.constants import (
    CHECKPOINTS_TABLE,
    DATASETS_TABLE,
    DATASET_RECORDS_TABLE,
    INFERENCE_REPORTS_TABLE,
    INFERENCE_RUNS_TABLE,
    TABLE_REQUIRED_COLUMNS,
    TRAINING_SAMPLES_TABLE,
    VALIDATION_IMAGE_STATS_TABLE,
    VALIDATION_PIXEL_DISTRIBUTION_TABLE,
    VALIDATION_TEXT_SUMMARY_TABLE,
)


UPSERT_CONFLICT_COLUMNS: dict[str, tuple[str, ...]] = {
    DATASETS_TABLE: ("name",),
    DATASET_RECORDS_TABLE: ("dataset_id", "image_name", "report_text"),
    TRAINING_SAMPLES_TABLE: ("processing_run_id", "record_id"),
    VALIDATION_TEXT_SUMMARY_TABLE: ("validation_run_id",),
    VALIDATION_IMAGE_STATS_TABLE: ("validation_run_id", "record_id"),
    VALIDATION_PIXEL_DISTRIBUTION_TABLE: ("validation_run_id", "bin"),
    CHECKPOINTS_TABLE: ("name",),
    INFERENCE_RUNS_TABLE: ("request_id",),
    INFERENCE_REPORTS_TABLE: ("inference_run_id", "input_image_name"),
}

SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ALLOWED_TABLE_NAMES = frozenset(TABLE_REQUIRED_COLUMNS.keys())


# -----------------------------------------------------------------------------
def is_string_object_column(series: pd.Series) -> bool:
    if not pd.api.types.is_object_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    return bool(non_null.map(lambda value: isinstance(value, str)).all())


# -----------------------------------------------------------------------------
def validate_sql_identifier(identifier: str) -> str:
    normalized = str(identifier or "").strip()
    if not normalized:
        raise ValueError("SQL identifier cannot be empty")
    if not SQL_IDENTIFIER_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid SQL identifier: {identifier}")
    return normalized


# -----------------------------------------------------------------------------
def validate_table_name(table_name: str) -> str:
    normalized = validate_sql_identifier(table_name)
    if normalized not in ALLOWED_TABLE_NAMES:
        raise ValueError(f"Unsupported table name: {table_name}")
    return normalized


# -----------------------------------------------------------------------------
def normalize_postgres_engine(engine: str | None) -> str:
    if not engine:
        return "postgresql+psycopg"
    lowered = engine.lower()
    if lowered in {"postgres", "postgresql"}:
        return "postgresql+psycopg"
    return engine


# -----------------------------------------------------------------------------
def resolve_conflict_columns(
    table_name: str,
    payload_columns: list[str],
) -> tuple[list[str], list[str]]:
    configured = list(UPSERT_CONFLICT_COLUMNS.get(table_name, ()))
    if not configured:
        return [], []
    missing_columns = [column for column in configured if column not in payload_columns]
    if missing_columns:
        return [], missing_columns
    return configured, []


# -----------------------------------------------------------------------------
def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    string_columns = [
        column
        for column in df.columns
        if pd.api.types.is_string_dtype(df[column])
        or is_string_object_column(df[column])
    ]
    if not string_columns:
        return df
    normalized = df.copy()
    for column in string_columns:
        object_series = normalized[column].astype(object)
        normalized[column] = object_series.where(object_series.notna(), None)
    return normalized


# -----------------------------------------------------------------------------
def validate_unique_key_values(
    records: list[dict[Any, Any]],
    unique_columns: list[str],
    table_name: str,
) -> None:
    if not unique_columns:
        return

    for unique_column in unique_columns:
        for index, record in enumerate(records):
            value = record.get(unique_column)
            if value is None:
                raise ValueError(
                    f"Missing value for conflict column '{unique_column}' "
                    f"in table '{table_name}' at record index {index}"
                )
            try:
                if pd.isna(value):
                    raise ValueError(
                        f"Invalid NaN value for conflict column '{unique_column}' "
                        f"in table '{table_name}' at record index {index}"
                    )
            except TypeError:
                continue
