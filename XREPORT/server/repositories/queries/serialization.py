from __future__ import annotations

from typing import Any

from XREPORT.server.common.constants import (
    CHECKPOINTS_TABLE,
    CHECKPOINT_EVALUATIONS_TABLE,
    DATASETS_TABLE,
    DATASET_RECORDS_TABLE,
    INFERENCE_RUNS_TABLE,
    PROCESSING_RUNS_TABLE,
    TRAINING_SAMPLES_TABLE,
    VALIDATION_IMAGE_STATS_TABLE,
    VALIDATION_PIXEL_DISTRIBUTION_TABLE,
    VALIDATION_RUNS_TABLE,
    VALIDATION_TEXT_SUMMARY_TABLE,
)


# -----------------------------------------------------------------------------
def select_dataset_id_by_name_sql() -> str:
    return f'SELECT dataset_id FROM "{DATASETS_TABLE}" WHERE name = :name'


# -----------------------------------------------------------------------------
def select_checkpoint_id_by_name_sql() -> str:
    return f'SELECT checkpoint_id FROM "{CHECKPOINTS_TABLE}" WHERE name = :name'


# -----------------------------------------------------------------------------
def delete_by_key_sql(table_name: str, column_name: str) -> str:
    return f'DELETE FROM "{table_name}" WHERE {column_name} = :value'


# -----------------------------------------------------------------------------
def load_source_dataset_sql(
    dataset_name_filter: str | None = None,
) -> tuple[str, dict[str, Any]]:
    where_clause = ""
    parameters: dict[str, Any] = {}
    if dataset_name_filter:
        where_clause = "WHERE d.name = :dataset_name"
        parameters["dataset_name"] = dataset_name_filter
    sql = f'''
        SELECT
            d.dataset_id,
            d.name AS name,
            r.record_id,
            r.image_name AS image,
            r.report_text AS text,
            r.image_path AS path,
            r.row_order
        FROM "{DATASET_RECORDS_TABLE}" r
        JOIN "{DATASETS_TABLE}" d ON d.dataset_id = r.dataset_id
        {where_clause}
        ORDER BY d.name, r.row_order, r.record_id
    '''
    return sql, parameters


# -----------------------------------------------------------------------------
def latest_processing_run_sql(
    dataset_name_filter: str | None = None,
) -> tuple[str, dict[str, Any]]:
    where_clause = ""
    parameters: dict[str, Any] = {}
    if dataset_name_filter:
        where_clause = "WHERE d.name = :dataset_name"
        parameters["dataset_name"] = dataset_name_filter
    sql = f'''
        SELECT
            pr.processing_run_id,
            pr.dataset_id,
            pr.source_dataset_id,
            pr.config_hash,
            pr.executed_at,
            pr.seed,
            pr.sample_size,
            pr.validation_size,
            pr.split_seed,
            pr.vocabulary_size,
            pr.max_report_size,
            pr.tokenizer,
            d.name AS dataset_name,
            sd.name AS source_dataset
        FROM "{PROCESSING_RUNS_TABLE}" pr
        JOIN "{DATASETS_TABLE}" d ON d.dataset_id = pr.dataset_id
        LEFT JOIN "{DATASETS_TABLE}" sd ON sd.dataset_id = pr.source_dataset_id
        {where_clause}
        ORDER BY pr.processing_run_id DESC
        LIMIT 1
    '''
    return sql, parameters


# -----------------------------------------------------------------------------
def training_samples_for_processing_run_sql() -> str:
    return f'''
        SELECT
            ts.training_sample_id,
            dr.record_id,
            ts.split,
            ts.tokens_json AS tokens,
            dr.image_name AS image,
            dr.report_text AS text,
            dr.image_path AS path
        FROM "{TRAINING_SAMPLES_TABLE}" ts
        JOIN "{DATASET_RECORDS_TABLE}" dr ON dr.record_id = ts.record_id
        WHERE ts.processing_run_id = :processing_run_id
        ORDER BY ts.training_sample_id
    '''


# -----------------------------------------------------------------------------
def latest_processing_run_id_sql() -> str:
    return f'''
        SELECT
            processing_run_id
        FROM "{PROCESSING_RUNS_TABLE}"
        WHERE config_hash = :config_hash
            AND dataset_id = :dataset_id
        ORDER BY processing_run_id DESC
        LIMIT 1
    '''


# -----------------------------------------------------------------------------
def dataset_record_ids_sql() -> str:
    return f'''
        SELECT
            record_id
        FROM "{DATASET_RECORDS_TABLE}"
        WHERE dataset_id = :dataset_id
    '''


# -----------------------------------------------------------------------------
def select_inference_run_id_by_request_id_sql() -> str:
    return f'SELECT inference_run_id FROM "{INFERENCE_RUNS_TABLE}" WHERE request_id = :request_id'


# -----------------------------------------------------------------------------
def latest_validation_run_id_sql() -> str:
    return f'''
        SELECT
            validation_run_id
        FROM "{VALIDATION_RUNS_TABLE}"
        WHERE dataset_id = :dataset_id
        ORDER BY validation_run_id DESC
        LIMIT 1
    '''


# -----------------------------------------------------------------------------
def latest_validation_run_sql() -> str:
    return f'''
        SELECT
            validation_run_id,
            executed_at,
            sample_size,
            metrics_json,
            artifacts_json
        FROM "{VALIDATION_RUNS_TABLE}"
        WHERE dataset_id = :dataset_id
        ORDER BY validation_run_id DESC
        LIMIT 1
    '''


# -----------------------------------------------------------------------------
def validation_text_summary_sql() -> str:
    return f'''
        SELECT
            count,
            total_words,
            unique_words,
            avg_words_per_report,
            min_words_per_report,
            max_words_per_report
        FROM "{VALIDATION_TEXT_SUMMARY_TABLE}"
        WHERE validation_run_id = :validation_run_id
    '''


# -----------------------------------------------------------------------------
def validation_image_aggregate_sql() -> str:
    return f'''
        SELECT
            COUNT(*) AS count,
            AVG(height) AS mean_height,
            AVG(width) AS mean_width,
            AVG(mean) AS mean_pixel_value,
            AVG(std) AS std_pixel_value,
            AVG(noise_std) AS mean_noise_std,
            AVG(noise_ratio) AS mean_noise_ratio
        FROM "{VALIDATION_IMAGE_STATS_TABLE}"
        WHERE validation_run_id = :validation_run_id
    '''


# -----------------------------------------------------------------------------
def validation_pixel_distribution_sql() -> str:
    return f'''
        SELECT
            bin,
            count
        FROM "{VALIDATION_PIXEL_DISTRIBUTION_TABLE}"
        WHERE validation_run_id = :validation_run_id
        ORDER BY bin
    '''


# -----------------------------------------------------------------------------
def validation_report_exists_sql() -> str:
    return f'SELECT 1 FROM "{VALIDATION_RUNS_TABLE}" WHERE dataset_id = :dataset_id'


# -----------------------------------------------------------------------------
def latest_checkpoint_evaluation_sql() -> str:
    return f'''
        SELECT
            ce.executed_at,
            ce.metrics_json,
            ce.metric_configs_json,
            ce.results_json
        FROM "{CHECKPOINT_EVALUATIONS_TABLE}" ce
        JOIN "{CHECKPOINTS_TABLE}" c ON c.checkpoint_id = ce.checkpoint_id
        WHERE c.name = :checkpoint
        ORDER BY ce.evaluation_id DESC
        LIMIT 1
    '''


# -----------------------------------------------------------------------------
def checkpoint_evaluation_exists_sql() -> str:
    return f'''
        SELECT 1
        FROM "{CHECKPOINT_EVALUATIONS_TABLE}" ce
        JOIN "{CHECKPOINTS_TABLE}" c ON c.checkpoint_id = ce.checkpoint_id
        WHERE c.name = :checkpoint
    '''
