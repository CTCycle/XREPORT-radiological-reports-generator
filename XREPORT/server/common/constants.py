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
RADIOGRAPHY_TABLE = "radiography_data"
TRAINING_DATASET_TABLE = "training_dataset"
PROCESSING_METADATA_TABLE = "processing_metadata"
GENERATED_REPORTS_TABLE = "generated_reports"
TEXT_STATISTICS_TABLE = "text_statistics"
IMAGE_STATISTICS_TABLE = "image_statistics"
VALIDATION_REPORTS_TABLE = "validation_reports"
CHECKPOINT_EVALUATION_REPORTS_TABLE = "checkpoint_evaluation_reports"

TABLE_REQUIRED_COLUMNS: dict[str, list[str]] = {
    RADIOGRAPHY_TABLE: ["name", "image", "text", "path"],
    TRAINING_DATASET_TABLE: [
        "name",
        "hashcode",
        "image",
        "tokens",
        "split",
        "path",
    ],
    PROCESSING_METADATA_TABLE: [
        "name",
        "hashcode",
        "source_dataset",
        "date",
        "seed",
        "sample_size",
        "validation_size",
        "vocabulary_size",
        "max_report_size",
        "tokenizer",
    ],
    GENERATED_REPORTS_TABLE: ["image", "report", "checkpoint"],
    TEXT_STATISTICS_TABLE: ["name", "words_count"],
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
    VALIDATION_REPORTS_TABLE: [
        "name",
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
    RADIOGRAPHY_TABLE: ["name", "image"],
    TRAINING_DATASET_TABLE: ["name", "hashcode", "image", "split"],
    PROCESSING_METADATA_TABLE: ["hashcode"],
    GENERATED_REPORTS_TABLE: ["image", "checkpoint"],
    TEXT_STATISTICS_TABLE: ["name"],
    IMAGE_STATISTICS_TABLE: ["dataset_name", "name"],
    VALIDATION_REPORTS_TABLE: ["name"],
    CHECKPOINT_EVALUATION_REPORTS_TABLE: ["checkpoint"],
}
