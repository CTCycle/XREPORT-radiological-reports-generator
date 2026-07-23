from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
RESOURCES_DIR = APP_DIR / "resources"
SCRIPTS_DIR = APP_DIR / "scripts"
SERVER_DIR = APP_DIR / "server"
SETTINGS_DIR = ROOT_DIR / "settings"
SHARED_DIR = APP_DIR / "shared"
TESTS_DIR = APP_DIR / "tests"
XREPORT_DIR = APP_DIR / "XREPORT"

LOGS_DIR = RESOURCES_DIR / "logs"
MODELS_DIR = RESOURCES_DIR / "models"
CHECKPOINTS_DIR = RESOURCES_DIR / "checkpoints"
TEMPLATES_DIR = RESOURCES_DIR / "templates"
ENCODERS_DIR = MODELS_DIR / "XRAYEncoder"
TOKENIZERS_DIR = MODELS_DIR / "tokenizers"

CONFIGURATION_FILE_PATH = SETTINGS_DIR / "configurations.json"
DATABASE_FILE_PATH = RESOURCES_DIR / "database.db"
ENV_FILE_PATH = SETTINGS_DIR / ".env"
