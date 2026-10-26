from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .runtime_environment import get_runtime_environment_settings

###############################################################################
@dataclass(frozen=True)
class DatabaseSettings:
    backend: Literal["sqlite", "postgresql"]
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
    sqlite_path: Path | None = None

###############################################################################
@dataclass(frozen=True)
class GlobalSettings:
    seed: int

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
    model_idle_timeout: int

###############################################################################
@dataclass(frozen=True)
class ServerSettings:
    database: DatabaseSettings
    global_settings: GlobalSettings
    jobs: JobsSettings
    inference: InferenceSettings

###############################################################################
class ApplicationSettingsValues(BaseModel):
    """Strict settings that users can change through the application UI."""

    model_config = ConfigDict(extra="forbid", strict=True)

    global_seed: int = Field(ge=0, le=4_294_967_295)
    job_polling_interval: float = Field(
        ge=0.25,
        le=60.0,
        allow_inf_nan=False,
    )
    inference_model_timeout: int = Field(ge=1)
    inference_model_idle_timeout: int = Field(ge=0)
    theme_preference: Literal["light", "dark", "system"] | None = None


DEFAULT_APPLICATION_SETTINGS = ApplicationSettingsValues(
    global_seed=42,
    job_polling_interval=1.0,
    inference_model_timeout=600,
    inference_model_idle_timeout=300,
    theme_preference=None,
)

###############################################################################
def application_settings_to_server_settings(
    values: ApplicationSettingsValues,
    database: DatabaseSettings,
) -> ServerSettings:
    runtime = get_runtime_environment_settings()
    return ServerSettings(
        database=database,
        global_settings=GlobalSettings(seed=values.global_seed),
        jobs=JobsSettings(polling_interval=values.job_polling_interval),
        inference=InferenceSettings(
            hf_local_only=runtime.inference_hf_local_only,
            device=runtime.inference_device,
            model_timeout=values.inference_model_timeout,
            model_idle_timeout=values.inference_model_idle_timeout,
        ),
    )
