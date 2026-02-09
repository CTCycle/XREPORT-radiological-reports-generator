from __future__ import annotations

from typing import Any

import pandas as pd

from XREPORT.server.common.constants import (
    IMAGE_STATISTICS_TABLE,
    PROCESSING_METADATA_TABLE,
    RADIOGRAPHY_TABLE,
    TRAINING_DATASET_TABLE,
)


UPSERT_CONFLICT_COLUMNS: dict[str, tuple[str, ...]] = {
    RADIOGRAPHY_TABLE: ("name", "image", "text"),
    TRAINING_DATASET_TABLE: ("hashcode", "image", "text"),
    PROCESSING_METADATA_TABLE: ("hashcode",),
    IMAGE_STATISTICS_TABLE: ("dataset_name", "name"),
}


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
        if pd.api.types.is_string_dtype(df[column].dtype)
    ]
    if not string_columns:
        return df
    normalized = df.copy()
    normalized[string_columns] = normalized[string_columns].astype(object)
    normalized[string_columns] = normalized[string_columns].apply(
        lambda column: column.map(lambda value: None if pd.isna(value) else value)
    )
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
