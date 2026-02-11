from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../../.."))
PROJECT_DIR = join(ROOT_DIR, "XREPORT")
SETTING_PATH = join(PROJECT_DIR, "settings")
RESOURCES_PATH = join(PROJECT_DIR, "resources")
LOGS_PATH = join(RESOURCES_PATH, "logs")
ENV_FILE_PATH = join(SETTING_PATH, ".env")
MODELS_PATH = join(RESOURCES_PATH, "models")
ENCODERS_PATH = join(MODELS_PATH, "XRAYEncoder")
TOKENIZERS_PATH = join(MODELS_PATH, "tokenizers")
CHECKPOINT_PATH = join(RESOURCES_PATH, "checkpoints")
DATABASE_FILENAME = "database.db"

###############################################################################
CONFIGURATION_FILE = join(SETTING_PATH, "configurations.json")

# [FASTAPI METADATA]
###############################################################################
FASTAPI_TITLE = "XREPORT Backend"
FASTAPI_DESCRIPTION = "FastAPI backend"
FASTAPI_VERSION = "1.0.0"

# [ENDPOINS]
###############################################################################
BASE_URL = "/base/tags"


# [TRAINING CONSTANTS]
###############################################################################
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


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

