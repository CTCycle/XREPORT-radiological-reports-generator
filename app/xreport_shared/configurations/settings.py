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
    max_new_tokens_deterministic: int = 0
    max_new_tokens_concise: int = 0
    max_new_tokens_detailed: int = 0
    detailed_num_beams: int = 0


###############################################################################
@dataclass(frozen=True)
class ServerSettings:
    database: DatabaseSettings
    global_settings: GlobalSettings
    jobs: JobsSettings
    # Only XREPORT Inference persists inference controls; Studio has none.
    inference: InferenceSettings | None


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
    theme_preference: Literal["light", "dark", "system"] | None = None


###############################################################################
class InferenceSettingsValues(ApplicationSettingsValues):
    """XREPORT Inference settings, adding local inference operation limits."""

    inference_model_timeout: int = Field(ge=1)
    inference_model_idle_timeout: int = Field(ge=0)
    # 0 defers to the selected model adapter's own profile default.
    inference_max_new_tokens_deterministic: int = Field(default=0, ge=0, le=4096)
    inference_max_new_tokens_concise: int = Field(default=0, ge=0, le=4096)
    inference_max_new_tokens_detailed: int = Field(default=0, ge=0, le=4096)
    inference_detailed_num_beams: int = Field(default=0, ge=0, le=8)


DEFAULT_STUDIO_SETTINGS = ApplicationSettingsValues(
    global_seed=42,
    job_polling_interval=1.0,
    theme_preference=None,
)
DEFAULT_INFERENCE_SETTINGS = InferenceSettingsValues(
    global_seed=42,
    job_polling_interval=1.0,
    inference_model_timeout=600,
    inference_model_idle_timeout=300,
    inference_max_new_tokens_deterministic=0,
    inference_max_new_tokens_concise=0,
    inference_max_new_tokens_detailed=0,
    inference_detailed_num_beams=0,
    theme_preference=None,
)


###############################################################################
def default_application_settings(profile: str) -> ApplicationSettingsValues:
    return DEFAULT_STUDIO_SETTINGS if profile == "studio" else DEFAULT_INFERENCE_SETTINGS


###############################################################################
def application_settings_to_server_settings(
    values: ApplicationSettingsValues,
    database: DatabaseSettings,
) -> ServerSettings:
    inference: InferenceSettings | None = None
    if isinstance(values, InferenceSettingsValues):
        runtime = get_runtime_environment_settings()
        inference = InferenceSettings(
            hf_local_only=runtime.inference_hf_local_only,
            device=runtime.inference_device,
            model_timeout=values.inference_model_timeout,
            model_idle_timeout=values.inference_model_idle_timeout,
            max_new_tokens_deterministic=values.inference_max_new_tokens_deterministic,
            max_new_tokens_concise=values.inference_max_new_tokens_concise,
            max_new_tokens_detailed=values.inference_max_new_tokens_detailed,
            detailed_num_beams=values.inference_detailed_num_beams,
        )
    return ServerSettings(
        database=database,
        global_settings=GlobalSettings(seed=values.global_seed),
        jobs=JobsSettings(polling_interval=values.job_polling_interval),
        inference=inference,
    )
