from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

###############################################################################
class _StrictSettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

###############################################################################
class RuntimeGlobalSettings(_StrictSettingsModel):
    seed: int

###############################################################################
class RuntimeJobSettings(_StrictSettingsModel):
    polling_interval: float

###############################################################################
class RuntimeInferenceSettings(_StrictSettingsModel):
    model_timeout: int
    model_idle_timeout: int

###############################################################################
class RuntimeApplicationSettings(_StrictSettingsModel):
    global_settings: RuntimeGlobalSettings = Field(alias="global")
    jobs: RuntimeJobSettings
    inference: RuntimeInferenceSettings
    theme_preference: Literal["light", "dark", "system"] | None

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        populate_by_name=False,
    )

###############################################################################
class ApplicationSettingsResponse(_StrictSettingsModel):
    values: RuntimeApplicationSettings
    defaults: RuntimeApplicationSettings

###############################################################################
class _PatchModel(_StrictSettingsModel):

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def reject_explicit_nulls(self) -> _PatchModel:
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be null")
        if not self.model_fields_set:
            raise ValueError("at least one setting must be provided")
        return self

###############################################################################
class GlobalSettingsPatch(_PatchModel):
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)

###############################################################################
class JobSettingsPatch(_PatchModel):
    polling_interval: float | None = Field(
        default=None,
        ge=0.25,
        le=60.0,
        allow_inf_nan=False,
    )

###############################################################################
class InferenceSettingsPatch(_PatchModel):
    model_timeout: int | None = Field(default=None, ge=1)
    model_idle_timeout: int | None = Field(default=None, ge=0)

###############################################################################
class ApplicationSettingsPatch(_StrictSettingsModel):
    global_settings: GlobalSettingsPatch | None = Field(default=None, alias="global")
    jobs: JobSettingsPatch | None = None
    inference: InferenceSettingsPatch | None = None
    theme_preference: Literal["light", "dark", "system"] | None = None

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        populate_by_name=False,
    )

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_patch(self) -> ApplicationSettingsPatch:
        if not self.model_fields_set:
            raise ValueError("at least one setting must be provided")
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be null")
        return self
