from __future__ import annotations

from pathlib import Path

# [PATHS]
###############################################################################
ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
SETTING_PATH = ROOT_DIR / "settings"
RESOURCES_PATH = APP_DIR / "resources"
LOGS_PATH = RESOURCES_PATH / "logs"
ENV_FILE_PATH = SETTING_PATH / ".env"
MODELS_PATH = RESOURCES_PATH / "models"
ENCODERS_PATH = MODELS_PATH / "XRAYEncoder"
TOKENIZERS_PATH = MODELS_PATH / "tokenizers"
CHECKPOINT_PATH = RESOURCES_PATH / "checkpoints"
TEMPLATES_PATH = RESOURCES_PATH / "templates"
DATABASE_FILENAME = "database.db"
DATABASE_FILE_PATH = RESOURCES_PATH / DATABASE_FILENAME
CLIENT_DIST_PATH = APP_DIR / "client" / "dist"
CLIENT_ASSETS_PATH = CLIENT_DIST_PATH / "assets"
CLIENT_INDEX_FILE_PATH = CLIENT_DIST_PATH / "index.html"

###############################################################################
CONFIGURATION_FILE = SETTING_PATH / "configurations.json"

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
PROCESSING_RUNS_TABLE = "processing_runs"
TRAINING_SAMPLES_TABLE = "training_samples"
VALIDATION_RUNS_TABLE = "validation_runs"
VALIDATION_TEXT_SUMMARY_TABLE = "validation_text_summary"
VALIDATION_IMAGE_STATS_TABLE = "validation_image_stats"
VALIDATION_PIXEL_DISTRIBUTION_TABLE = "validation_pixel_distribution"
CHECKPOINTS_TABLE = "checkpoints"
CHECKPOINT_EVALUATIONS_TABLE = "checkpoint_evaluations"
INFERENCE_RUNS_TABLE = "inference_runs"
INFERENCE_REPORTS_TABLE = "inference_reports"

TABLE_REQUIRED_COLUMNS: dict[str, list[str]] = {
    DATASETS_TABLE: ["name", "created_at"],
    DATASET_RECORDS_TABLE: [
        "dataset_id",
        "image_name",
        "report_text",
        "image_path",
        "row_order",
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
    VALIDATION_TEXT_SUMMARY_TABLE: [
        "validation_run_id",
        "count",
        "total_words",
        "unique_words",
        "avg_words_per_report",
        "min_words_per_report",
        "max_words_per_report",
    ],
    VALIDATION_IMAGE_STATS_TABLE: [
        "validation_run_id",
        "record_id",
        "height",
        "width",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "pixel_range",
        "noise_std",
        "noise_ratio",
    ],
    VALIDATION_PIXEL_DISTRIBUTION_TABLE: [
        "validation_run_id",
        "bin",
        "count",
    ],
    CHECKPOINTS_TABLE: [
        "name",
        "path",
        "created_at",
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
        "executed_at",
    ],
    INFERENCE_REPORTS_TABLE: [
        "inference_run_id",
        "input_image_name",
        "generated_report",
    ],
}
