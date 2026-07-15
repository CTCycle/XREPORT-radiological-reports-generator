from __future__ import annotations

# [BACKEND ROUTES]
###############################################################################
FASTAPI_ROOT_ENDPOINT = "/"
FASTAPI_DOCS_ENDPOINT = "/docs"
FASTAPI_API_PREFIX = "/api"
FASTAPI_ASSETS_ENDPOINT = "/assets"
FASTAPI_SPA_FALLBACK_ENDPOINT = "/{full_path:path}"

# [FASTAPI METADATA]
###############################################################################
FASTAPI_TITLE = "XREPORT Backend"
FASTAPI_DESCRIPTION = "FastAPI backend"
FASTAPI_VERSION = "1.0.0"

# [TRAINING CONSTANTS]
###############################################################################
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
INFERENCE_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
INFERENCE_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
    "image/tiff",
}


# [DATABASE TABLES]
###############################################################################
DATASETS_TABLE = "datasets"
DATASET_RECORDS_TABLE = "dataset_records"
DATASET_VERSIONS_TABLE = "dataset_versions"
PROCESSING_RUNS_TABLE = "processing_runs"
TRAINING_SAMPLES_TABLE = "training_samples"
VALIDATION_RUNS_TABLE = "validation_runs"
CHECKPOINTS_TABLE = "checkpoints"
CHECKPOINT_EVALUATIONS_TABLE = "checkpoint_evaluations"
INFERENCE_RUNS_TABLE = "inference_runs"
INFERENCE_REPORTS_TABLE = "inference_reports"

TABLE_REQUIRED_COLUMNS: dict[str, list[str]] = {
    DATASETS_TABLE: ["name", "created_at"],
    DATASET_RECORDS_TABLE: [
        "dataset_id",
        "dataset_version_id",
        "image_name",
        "report_text",
        "image_path",
        "row_order",
    ],
    DATASET_VERSIONS_TABLE: [
        "dataset_id",
        "version_number",
        "content_hash",
        "record_count",
        "imported_at",
    ],
    PROCESSING_RUNS_TABLE: [
        "dataset_id",
        "config_hash",
        "executed_at",
        "seed",
        "sample_size",
        "validation_size",
        "split_seed",
        "vocabulary_size",
        "max_report_size",
        "tokenizer",
    ],
    TRAINING_SAMPLES_TABLE: [
        "processing_run_id",
        "record_id",
        "split",
        "tokens_json",
    ],
    VALIDATION_RUNS_TABLE: [
        "dataset_id",
        "executed_at",
        "sample_size",
        "metrics_json",
    ],
    CHECKPOINTS_TABLE: [
        "name",
        "name_key",
        "path",
        "created_at",
        "last_seen_at",
    ],
    CHECKPOINT_EVALUATIONS_TABLE: [
        "checkpoint_id",
        "executed_at",
        "metrics_json",
        "metric_configs_json",
        "results_json",
    ],
    INFERENCE_RUNS_TABLE: [
        "checkpoint_id",
        "generation_mode",
        "request_id",
        "status",
        "executed_at",
    ],
    INFERENCE_REPORTS_TABLE: [
        "inference_run_id",
        "input_image_name",
        "input_image_name_key",
        "image_index",
        "generated_report",
    ],
}
