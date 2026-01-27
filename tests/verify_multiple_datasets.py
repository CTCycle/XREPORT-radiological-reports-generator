from __future__ import annotations

import datetime
import uuid

import pandas as pd
import sqlalchemy

from XREPORT.server.database.database import database


def build_dataset(dataset_name: str, rows: int = 3) -> pd.DataFrame:
    records = []
    for idx in range(1, rows + 1):
        records.append(
            {
                "dataset_name": dataset_name,
                "id": idx,
                "image": f"{dataset_name}_image_{idx}",
                "text": f"Sample report {idx} for {dataset_name}",
                "path": f"C:/tmp/{dataset_name}_image_{idx}.png",
            }
        )
    return pd.DataFrame(
        records,
        columns=["dataset_name", "id", "image", "text", "path"],
    )


def main() -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    nonce = uuid.uuid4().hex[:8]
    dataset_a = f"test_dataset_a_{timestamp}_{nonce}"
    dataset_b = f"test_dataset_b_{timestamp}_{nonce}"

    df_a = build_dataset(dataset_a, rows=2)
    df_b = build_dataset(dataset_b, rows=3)

    initial_count = database.count_rows("RADIOGRAPHY_DATA")

    database.upsert_into_database(df_a, "RADIOGRAPHY_DATA")
    database.upsert_into_database(df_b, "RADIOGRAPHY_DATA")

    after_count = database.count_rows("RADIOGRAPHY_DATA")
    expected_increase = len(df_a) + len(df_b)
    if after_count != initial_count + expected_increase:
        raise RuntimeError(
            "Unexpected row count after append. "
            f"Expected +{expected_increase}, got {after_count - initial_count}."
        )

    with database.backend.engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.text(
                'SELECT DISTINCT dataset_name FROM "RADIOGRAPHY_DATA" '
                "WHERE dataset_name IN (:a, :b) ORDER BY dataset_name"
            ),
            {"a": dataset_a, "b": dataset_b},
        )
        dataset_names = {row[0] for row in result.fetchall()}

    if dataset_a not in dataset_names or dataset_b not in dataset_names:
        raise RuntimeError(
            "Dataset names not found after append. "
            f"Found: {sorted(dataset_names)}."
        )

    with database.backend.engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                'DELETE FROM "RADIOGRAPHY_DATA" WHERE dataset_name IN (:a, :b)'
            ),
            {"a": dataset_a, "b": dataset_b},
        )

    final_count = database.count_rows("RADIOGRAPHY_DATA")
    if final_count != initial_count:
        raise RuntimeError(
            "Cleanup did not restore original row count. "
            f"Expected {initial_count}, got {final_count}."
        )

    print(
        "Multiple dataset append verification passed. "
        f"Inserted {expected_increase} rows and confirmed dataset names."
    )


if __name__ == "__main__":
    main()
