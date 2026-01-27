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

# [ENDPOINS]
###############################################################################
BASE_URL = "/base/tags"






# [TRAINING CONSTANTS]
###############################################################################
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


