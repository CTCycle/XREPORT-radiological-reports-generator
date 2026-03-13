from __future__ import annotations

from XREPORT.server.common.constants import (
    DATASETS_TABLE,
    DATASET_RECORDS_TABLE,
    PROCESSING_RUNS_TABLE,
    TRAINING_SAMPLES_TABLE,
    VALIDATION_RUNS_TABLE,
)


# -----------------------------------------------------------------------------
def dataset_names_sql() -> str:
    return f"""
        SELECT
            d.name,
            MIN(r.image_path) AS sample_path,
            COUNT(r.record_id) AS row_count,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM "{VALIDATION_RUNS_TABLE}" vr
                    WHERE vr.dataset_id = d.dataset_id
                ) THEN TRUE
                ELSE FALSE
            END AS has_validation_report
        FROM "{DATASETS_TABLE}" d
        JOIN "{DATASET_RECORDS_TABLE}" r ON r.dataset_id = d.dataset_id
        GROUP BY d.dataset_id, d.name
        ORDER BY d.name
    """


# -----------------------------------------------------------------------------
def processed_dataset_names_sql() -> str:
    return f"""
        WITH latest_runs AS (
            SELECT
                dataset_id,
                MAX(processing_run_id) AS processing_run_id
            FROM "{PROCESSING_RUNS_TABLE}"
            GROUP BY dataset_id
        )
        SELECT
            d.name,
            COUNT(ts.training_sample_id) AS row_count,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM "{VALIDATION_RUNS_TABLE}" vr
                    WHERE vr.dataset_id = d.dataset_id
                ) THEN TRUE
                ELSE FALSE
            END AS has_validation_report
        FROM latest_runs lr
        JOIN "{DATASETS_TABLE}" d ON d.dataset_id = lr.dataset_id
        LEFT JOIN "{TRAINING_SAMPLES_TABLE}" ts ON ts.processing_run_id = lr.processing_run_id
        GROUP BY d.dataset_id, d.name
        ORDER BY d.name
    """


# -----------------------------------------------------------------------------
def delete_dataset_by_name_sql() -> str:
    return f'DELETE FROM "{DATASETS_TABLE}" WHERE name = :dataset_name'


# -----------------------------------------------------------------------------
def dataset_image_count_sql() -> str:
    return f'''
        SELECT COUNT(*)
        FROM "{DATASET_RECORDS_TABLE}" r
        JOIN "{DATASETS_TABLE}" d ON d.dataset_id = r.dataset_id
        WHERE d.name = :dataset_name
    '''


# -----------------------------------------------------------------------------
def dataset_image_metadata_sql() -> str:
    return f'''
        SELECT r.image_name, r.report_text, r.image_path
        FROM "{DATASET_RECORDS_TABLE}" r
        JOIN "{DATASETS_TABLE}" d ON d.dataset_id = r.dataset_id
        WHERE d.name = :dataset_name
        ORDER BY r.row_order, r.record_id
        LIMIT 1 OFFSET :offset
    '''


# -----------------------------------------------------------------------------
def dataset_image_content_sql() -> str:
    return f'''
        SELECT r.image_path
        FROM "{DATASET_RECORDS_TABLE}" r
        JOIN "{DATASETS_TABLE}" d ON d.dataset_id = r.dataset_id
        WHERE d.name = :dataset_name
        ORDER BY r.row_order, r.record_id
        LIMIT 1 OFFSET :offset
    '''
