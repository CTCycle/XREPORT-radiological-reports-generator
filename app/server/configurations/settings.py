from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal

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
class ApplicationSettingsValues(BaseModel):
    """Strict persisted application settings, including private runtime policy."""

    model_config = ConfigDict(extra="forbid", strict=True)

    global_seed: int = Field(ge=0, le=4_294_967_295)
    allow_local_filesystem_access: bool
    job_polling_interval: float = Field(
        ge=0.25,
        le=60.0,
        allow_inf_nan=False,
    )
    inference_hf_local_only: bool
    inference_device: Literal["auto", "cpu", "cuda"]
    inference_model_timeout: int = Field(ge=1)


DEFAULT_APPLICATION_SETTINGS = ApplicationSettingsValues(
    global_seed=42,
    allow_local_filesystem_access=True,
    job_polling_interval=1.0,
    inference_hf_local_only=True,
    inference_device="auto",
    inference_model_timeout=600,
)

###############################################################################
def application_settings_to_server_settings(
    values: ApplicationSettingsValues,
    database: DatabaseSettings,
) -> ServerSettings:
    return ServerSettings(
        database=database,
        global_settings=GlobalSettings(seed=values.global_seed),
        features=FeatureSettings(
            allow_local_filesystem_access=values.allow_local_filesystem_access
        ),
        jobs=JobsSettings(polling_interval=values.job_polling_interval),
        inference=InferenceSettings(
            hf_local_only=values.inference_hf_local_only,
            device=values.inference_device,
            model_timeout=values.inference_model_timeout,
        ),
    )


###############################################################################
def database_settings_from_environment() -> DatabaseSettings:
    return _database_env_settings()
