from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../../.."))
PROJECT_DIR = join(ROOT_DIR, "APP")
SETTING_PATH = join(PROJECT_DIR, "settings")
RESOURCES_PATH = join(PROJECT_DIR, "resources")
DATA_PATH = join(RESOURCES_PATH, "database")
LOGS_PATH = join(RESOURCES_PATH, "logs")
ENV_FILE_PATH = join(SETTING_PATH, ".env")
DATABASE_FILENAME = "sqlite.db"

###############################################################################
SERVER_CONFIGURATION_FILE = join(SETTING_PATH, "server_configurations.json")

# [ENDPOINS]
###############################################################################
BASE_URL = "/base/tags"

# [EXTERNAL DATA SOURCES]
###############################################################################
CONSTANT = 1.0

# [DATABASE TABLES]
###############################################################################
GEONAMES_TABLE = "TABLE"

# [DATABASE COLUMNS]
###############################################################################
TABLE_COLUMNS = [
    "A",
    "B",
    "C",
]