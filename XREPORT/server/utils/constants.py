from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../../.."))
PROJECT_DIR = join(ROOT_DIR, "XREPORT")
SETTING_PATH = join(PROJECT_DIR, "settings")
RESOURCES_PATH = join(PROJECT_DIR, "resources")
DATA_PATH = join(RESOURCES_PATH, "database")
LOGS_PATH = join(RESOURCES_PATH, "logs")
ENV_FILE_PATH = join(SETTING_PATH, ".env")
MODELS_PATH = join(RESOURCES_PATH, "models")
ENCODERS_PATH = join(MODELS_PATH, "XRAYEncoder")
TOKENIZERS_PATH = join(MODELS_PATH, "tokenizers")
CHECKPOINT_PATH = join(RESOURCES_PATH, "checkpoints")
DATABASE_FILENAME = "sqlite.db"

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
RADIOGRAPHY_TABLE = "RADIOGRAPHY_DATA"
TRAINING_DATASET_TABLE = "TRAINING_DATASET"
PROCESSING_METADATA_TABLE = "PROCESSING_METADATA"
GENERATED_REPORTS_TABLE = "GENERATED_REPORTS"
TEXT_STATISTICS_TABLE = "TEXT_STATISTICS"
IMAGE_STATISTICS_TABLE = "IMAGE_STATISTICS"
CHECKPOINTS_SUMMARY_TABLE = "CHECKPOINTS_SUMMARY"
VALIDATION_REPORTS_TABLE = "VALIDATION_REPORTS"
CHECKPOINT_EVALUATION_REPORTS_TABLE = "CHECKPOINT_EVALUATION_REPORTS"

TABLE_REQUIRED_COLUMNS: dict[str, list[str]] = {
    RADIOGRAPHY_TABLE: ["dataset_name", "id", "image", "text", "path"],
    TRAINING_DATASET_TABLE: [
        "dataset_name",
        "hashcode",
        "id",
        "image",
        "tokens",
        "split",
        "path",
    ],
    PROCESSING_METADATA_TABLE: [
        "dataset_name",
        "hashcode",
        "id",
        "date",
        "seed",
        "sample_size",
        "validation_size",
        "vocabulary_size",
        "max_report_size",
        "tokenizer",
    ],
    GENERATED_REPORTS_TABLE: ["image", "report", "checkpoint"],
    TEXT_STATISTICS_TABLE: ["dataset_name", "name", "words_count"],
    IMAGE_STATISTICS_TABLE: [
        "dataset_name",
        "name",
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
    CHECKPOINTS_SUMMARY_TABLE: ["checkpoint"],
    VALIDATION_REPORTS_TABLE: [
        "dataset_name",
        "date",
        "sample_size",
        "metrics",
        "text_statistics",
        "image_statistics",
        "pixel_distribution",
        "artifacts",
    ],
    CHECKPOINT_EVALUATION_REPORTS_TABLE: [
        "checkpoint",
        "date",
        "metrics",
        "metric_configs",
        "results",
    ],
}

TABLE_MERGE_KEYS: dict[str, list[str]] = {
    RADIOGRAPHY_TABLE: ["dataset_name", "id"],
    TRAINING_DATASET_TABLE: ["hashcode", "id"],
    PROCESSING_METADATA_TABLE: ["hashcode", "id"],
    GENERATED_REPORTS_TABLE: ["image", "checkpoint"],
    TEXT_STATISTICS_TABLE: ["dataset_name", "name"],
    IMAGE_STATISTICS_TABLE: ["dataset_name", "name"],
    CHECKPOINTS_SUMMARY_TABLE: ["checkpoint"],
    VALIDATION_REPORTS_TABLE: ["dataset_name"],
    CHECKPOINT_EVALUATION_REPORTS_TABLE: ["checkpoint"],
}
