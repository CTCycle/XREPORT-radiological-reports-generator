from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
SETTINGS_PATH = ROOT_DIR / "settings"
RESOURCES_PATH = APP_DIR / "resources"
LOGS_PATH = RESOURCES_PATH / "logs"
ENV_FILE_PATH = SETTINGS_PATH / ".env"
MODELS_PATH = RESOURCES_PATH / "models"
ENCODERS_PATH = MODELS_PATH / "XRAYEncoder"
TOKENIZERS_PATH = MODELS_PATH / "tokenizers"
CHECKPOINT_PATH = RESOURCES_PATH / "checkpoints"
TEMPLATES_PATH = RESOURCES_PATH / "templates"
DATABASE_FILE_PATH = RESOURCES_PATH / "database.db"
CLIENT_DIST_PATH = APP_DIR / "client" / "dist"
CLIENT_ASSETS_PATH = CLIENT_DIST_PATH / "assets"
CLIENT_INDEX_FILE_PATH = CLIENT_DIST_PATH / "index.html"
CONFIGURATION_FILE_PATH = SETTINGS_PATH / "configurations.json"
