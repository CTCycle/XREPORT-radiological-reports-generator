from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../../.."))
PROJECT_DIR = join(ROOT_DIR, "XREPORT")
RESOURCES_PATH = join(PROJECT_DIR, "resources")
DATA_PATH = join(RESOURCES_PATH, "database")
SOURCE_PATH = join(DATA_PATH, "dataset")
IMG_PATH = join(SOURCE_PATH, "images")
METADATA_PATH = join(DATA_PATH, "metadata")
EVALUATION_PATH = join(DATA_PATH, "validation")
MODELS_PATH = join(RESOURCES_PATH, "models")
TOKENIZERS_PATH = join(MODELS_PATH, "tokenizers")
ENCODERS_PATH = join(MODELS_PATH, "XRAYEncoder")
CHECKPOINT_PATH = join(RESOURCES_PATH, "checkpoints")
INFERENCE_INPUT_PATH = join(DATA_PATH, "inference")
CONFIG_PATH = join(RESOURCES_PATH, "configurations")
LOGS_PATH = join(RESOURCES_PATH, "logs")
PROCESS_METADATA_FILE = join(METADATA_PATH, "preprocessing_metadata.json")

# [UI LAYOUT PATH]
###############################################################################
UI_PATH = join(PROJECT_DIR, "app", "layout", "main_window.ui")
