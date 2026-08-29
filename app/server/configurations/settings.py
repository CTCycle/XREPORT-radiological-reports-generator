from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

###############################################################################
@dataclass(frozen=True)
class DatabaseSettings:
    backend: str
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
class InferenceSettings:
    hf_local_only: bool
    device: str
    max_loaded_models: int
    model_timeout: int

###############################################################################
@dataclass(frozen=True)
class ServerSettings:
    database: DatabaseSettings
    global_settings: GlobalSettings
    features: FeatureSettings
    jobs: JobsSettings
    inference: InferenceSettings

###############################################################################
def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

###############################################################################
def _parse_bool_env(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")

###############################################################################
def _normalize_int_env(
    name: str,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed

###############################################################################
def _required_env(name: str) -> str:
    value = _normalize_optional_string(os.getenv(name))
    if value is None:
        raise ValueError(f"{name} is required for external database mode")
    return value

###############################################################################
def _database_env_settings() -> DatabaseSettings:
    if _normalize_optional_string(os.getenv("DATABASE_URL")) is not None:
        raise ValueError(
            "DATABASE_URL is not supported; configure the decomposed DATABASE_* values"
        )
    if _parse_bool_env("EMBEDDED_DATABASE", default=True):
        return DatabaseSettings(
            backend="sqlite",
            engine=None,
            host=None,
            port=None,
            database_name=None,
            username=None,
            password=None,
            ssl=False,
            ssl_ca=None,
            connect_timeout=_normalize_int_env(
                "DATABASE_CONNECT_TIMEOUT", default=30, minimum=1
            ),
            insert_batch_size=_normalize_int_env(
                "DATABASE_INSERT_BATCH_SIZE", default=1000, minimum=1
            ),
        )

    return DatabaseSettings(
        backend="postgresql",
        engine=_required_env("DATABASE_ENGINE"),
        host=_required_env("DATABASE_HOST"),
        port=_normalize_int_env(
            "DATABASE_PORT", default=5432, minimum=1, maximum=65535
        ),
        database_name=_required_env("DATABASE_NAME"),
        username=_required_env("DATABASE_USERNAME"),
        password=_normalize_optional_string(os.getenv("DATABASE_PASSWORD")),
        ssl=_parse_bool_env("DATABASE_SSL", default=False),
        ssl_ca=_normalize_optional_string(os.getenv("DATABASE_SSL_CA")),
        connect_timeout=_normalize_int_env(
            "DATABASE_CONNECT_TIMEOUT", default=30, minimum=1
        ),
        insert_batch_size=_normalize_int_env(
            "DATABASE_INSERT_BATCH_SIZE", default=1000, minimum=1
        ),
    )

###############################################################################
class _StrictSettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=False)

###############################################################################
class JsonGlobalSettings(_StrictSettingsModel):
    seed: int = 42

###############################################################################
class JsonFeatureSettings(_StrictSettingsModel):
    allow_local_filesystem_access: bool = True

###############################################################################
class JsonJobsSettings(_StrictSettingsModel):
    polling_interval: float = 1.0

###############################################################################
class JsonInferenceSettings(_StrictSettingsModel):
    hf_local_only: bool = True
    device: str = "auto"
    max_loaded_models: int = Field(default=1, ge=1, le=1)
    model_timeout: int = Field(default=600, ge=1)

###############################################################################
class JsonServerSettings(_StrictSettingsModel):
    global_settings: JsonGlobalSettings = Field(
        default_factory=JsonGlobalSettings,
        alias="global",
    )
    features: JsonFeatureSettings = Field(default_factory=JsonFeatureSettings)
    jobs: JsonJobsSettings = Field(default_factory=JsonJobsSettings)
    inference: JsonInferenceSettings = Field(default_factory=JsonInferenceSettings)

    # -------------------------------------------------------------------------
    def to_server_settings(self, database: DatabaseSettings | None = None) -> ServerSettings:
        return ServerSettings(
            database=database or _database_env_settings(),
            global_settings=GlobalSettings(seed=self.global_settings.seed),
            features=FeatureSettings(
                allow_local_filesystem_access=self.features.allow_local_filesystem_access
            ),
            jobs=JobsSettings(polling_interval=self.jobs.polling_interval),
            inference=InferenceSettings(
                hf_local_only=self.inference.hf_local_only,
                device=self.inference.device,
                max_loaded_models=self.inference.max_loaded_models,
                model_timeout=self.inference.model_timeout,
            ),
        )
