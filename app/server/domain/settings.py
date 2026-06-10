from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


###############################################################################
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
def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


###############################################################################
def _normalize_bool_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


###############################################################################
def _normalize_int_env(
    value: str | None,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except (TypeError, ValueError):
        return default
    if minimum is not None and parsed < minimum:
        return default
    if maximum is not None and parsed > maximum:
        return default
    return parsed


###############################################################################
def _database_env_payload() -> dict[str, Any]:
    database_url = _normalize_optional_string(os.getenv("XREPORT_DATABASE_URL"))
    url_parts = urlsplit(database_url) if database_url else None
    url_username = url_parts.username if url_parts else None
    url_password = url_parts.password if url_parts else None
    url_host = url_parts.hostname if url_parts else None
    url_port = url_parts.port if url_parts and url_parts.port else None
    url_database_name = None
    if url_parts:
        url_database_name = _normalize_optional_string(url_parts.path.lstrip("/"))

    engine = _normalize_optional_string(os.getenv("XREPORT_DB_ENGINE"))
    if not engine and url_parts and url_parts.scheme:
        engine = url_parts.scheme

    return {
        "embedded_database": _normalize_bool_env(
            os.getenv("XREPORT_DB_EMBEDDED"),
            default=True,
        ),
        "engine": engine or "postgres",
        "host": _normalize_optional_string(os.getenv("XREPORT_DB_HOST")) or url_host,
        "port": _normalize_int_env(
            os.getenv("XREPORT_DB_PORT"),
            default=url_port or 5432,
            minimum=1,
            maximum=65535,
        ),
        "database_name": _normalize_optional_string(os.getenv("XREPORT_DB_NAME"))
        or url_database_name,
        "username": _normalize_optional_string(os.getenv("XREPORT_DB_USERNAME"))
        or _normalize_optional_string(url_username),
        "password": _normalize_optional_string(os.getenv("XREPORT_DB_PASSWORD"))
        or _normalize_optional_string(url_password),
        "ssl": _normalize_bool_env(
            os.getenv("XREPORT_DB_SSL"),
            default=False,
        ),
        "ssl_ca": _normalize_optional_string(os.getenv("XREPORT_DB_SSL_CA")),
        "connect_timeout": _normalize_int_env(
            os.getenv("XREPORT_DB_CONNECT_TIMEOUT"),
            default=30,
            minimum=1,
        ),
        "insert_batch_size": _normalize_int_env(
            os.getenv("XREPORT_DB_INSERT_BATCH_SIZE"),
            default=1000,
            minimum=1,
        ),
    }

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

    # -------------------------------------------------------------------------
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
        return _normalize_optional_string(value)

    # -------------------------------------------------------------------------
    @field_validator("engine", mode="before")
    @classmethod
    def normalize_engine(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        return text or "postgres"

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_external_database_requirements(self) -> "JsonDatabaseSettings":
        if self.embedded_database:
            return self

        missing: list[str] = []
        if not self.host:
            missing.append("XREPORT_DB_HOST or XREPORT_DATABASE_URL")
        if not self.database_name:
            missing.append("XREPORT_DB_NAME or XREPORT_DATABASE_URL")
        if not self.username:
            missing.append("XREPORT_DB_USERNAME or XREPORT_DATABASE_URL")

        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"External database mode requires environment variables: {joined}"
            )
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
class JsonServerSettings(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    database: JsonDatabaseSettings = Field(
        default_factory=lambda: JsonDatabaseSettings.model_validate(_database_env_payload())
    )
    global_settings: JsonGlobalSettings = Field(default_factory=JsonGlobalSettings)
    features: JsonFeatureSettings = Field(default_factory=JsonFeatureSettings)
    jobs: JsonJobsSettings = Field(default_factory=JsonJobsSettings)

    # "global" is reserved, map it explicitly.

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def map_global_alias(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        mapped = dict(value)
        if "global" in mapped and "global_settings" not in mapped:
            mapped["global_settings"] = mapped.get("global", {})
        mapped["database"] = _database_env_payload()
        return mapped

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

    # -------------------------------------------------------------------------
    def to_blocks(self) -> dict[str, dict[str, Any]]:
        return {
            "database": self.database.model_dump(),
            "global": self.global_settings.model_dump(),
            "features": self.features.model_dump(),
            "jobs": self.jobs.model_dump(),
        }
