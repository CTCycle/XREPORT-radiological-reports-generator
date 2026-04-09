from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from XREPORT.server.common.constants import CONFIGURATION_FILE


###############################################################################
@dataclass(frozen=True)
class DatabaseSettings:
    embedded_database: bool
    engine: str | None
    host: str | None
    port: int | None
    database_name: str | None
    username: str | None
    password: str | None
    ssl: bool
    ssl_ca: str | None
    connect_timeout: int
    insert_batch_size: int


###############################################################################
@dataclass(frozen=True)
class GlobalSettings:
    seed: int


@dataclass(frozen=True)
class FeatureSettings:
    allow_local_filesystem_access: bool


###############################################################################
@dataclass(frozen=True)
class JobsSettings:
    polling_interval: float


###############################################################################
@dataclass(frozen=True)
class ServerSettings:
    database: DatabaseSettings
    global_settings: GlobalSettings
    features: FeatureSettings
    jobs: JobsSettings


###############################################################################
class JsonDatabaseSettings(BaseModel):
    embedded_database: bool = True
    engine: str = "postgres"
    host: str | None = None
    port: int = Field(default=5432, ge=1, le=65535)
    database_name: str | None = None
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = Field(default=30, ge=1)
    insert_batch_size: int = Field(default=1000, ge=1)

    @field_validator(
        "host",
        "database_name",
        "username",
        "password",
        "ssl_ca",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("engine", mode="before")
    @classmethod
    def normalize_engine(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        return text or "postgres"

    @model_validator(mode="after")
    def validate_external_database_requirements(self) -> "JsonDatabaseSettings":
        if self.embedded_database:
            return self

        missing: list[str] = []
        if not self.host:
            missing.append("database.host")
        if not self.database_name:
            missing.append("database.database_name")
        if not self.username:
            missing.append("database.username")

        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"External database mode requires configuration keys: {joined}")
        return self


###############################################################################
class JsonGlobalSettings(BaseModel):
    seed: int = 42


###############################################################################
class JsonFeatureSettings(BaseModel):
    allow_local_filesystem_access: bool = True


###############################################################################
class JsonJobsSettings(BaseModel):
    polling_interval: float = 1.0


###############################################################################
class JsonConfigurationSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        raw_path = getattr(settings_cls, "_configuration_file", CONFIGURATION_FILE)
        self.configuration_file = Path(raw_path)

    # -------------------------------------------------------------------------
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    # -------------------------------------------------------------------------
    def __call__(self) -> dict[str, Any]:
        if not self.configuration_file.exists():
            raise RuntimeError(f"Configuration file not found: {self.configuration_file}")

        try:
            payload = json.loads(self.configuration_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load configuration from {self.configuration_file}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Configuration must be a JSON object.")

        return {
            "database": payload.get("database", {}),
            "global_settings": payload.get("global", {}),
            "features": payload.get("features", {}),
            "jobs": payload.get("jobs", {}),
        }


###############################################################################
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    _configuration_file: ClassVar[str] = CONFIGURATION_FILE

    database: JsonDatabaseSettings = Field(default_factory=JsonDatabaseSettings)
    global_settings: JsonGlobalSettings = Field(default_factory=JsonGlobalSettings)
    features: JsonFeatureSettings = Field(default_factory=JsonFeatureSettings)
    jobs: JsonJobsSettings = Field(default_factory=JsonJobsSettings)

    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = Field(default=8000, ge=1, le=65535)
    ui_host: str = "127.0.0.1"
    ui_port: int = Field(default=8001, ge=1, le=65535)
    vite_api_base_url: str = "/api"
    reload: bool = True
    optional_dependencies: bool = False
    mplbackend: str | None = None
    keras_backend: str | None = None
    xreport_tauri_mode: bool = False

    @field_validator(
        "fastapi_host",
        "ui_host",
        "vite_api_base_url",
        "mplbackend",
        "keras_backend",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        _ = dotenv_settings
        return (
            init_settings,
            env_settings,
            JsonConfigurationSettingsSource(settings_cls),
            file_secret_settings,
        )

    # -------------------------------------------------------------------------
    def to_server_settings(self) -> ServerSettings:
        db = self.database
        if db.embedded_database:
            database_settings = DatabaseSettings(
                embedded_database=True,
                engine=None,
                host=None,
                port=None,
                database_name=None,
                username=None,
                password=None,
                ssl=False,
                ssl_ca=None,
                connect_timeout=db.connect_timeout,
                insert_batch_size=db.insert_batch_size,
            )
        else:
            normalized_engine = db.engine.strip().lower()
            database_settings = DatabaseSettings(
                embedded_database=False,
                engine=normalized_engine,
                host=db.host,
                port=db.port,
                database_name=db.database_name,
                username=db.username,
                password=db.password,
                ssl=db.ssl,
                ssl_ca=db.ssl_ca,
                connect_timeout=db.connect_timeout,
                insert_batch_size=db.insert_batch_size,
            )

        return ServerSettings(
            database=database_settings,
            global_settings=GlobalSettings(seed=self.global_settings.seed),
            features=FeatureSettings(
                allow_local_filesystem_access=self.features.allow_local_filesystem_access
            ),
            jobs=JobsSettings(polling_interval=self.jobs.polling_interval),
        )
