from __future__ import annotations

from functools import lru_cache

from server.configurations.settings import (
    ApplicationSettingsValues,
    InferenceSettingsValues,
    default_application_settings,
)
from server.domain.settings import (
    ApplicationSettingsPatch,
    ApplicationSettingsResponse,
    RuntimeApplicationSettings,
    RuntimeGlobalSettings,
    RuntimeInferenceSettings,
    RuntimeJobSettings,
)
from server.repositories.application_settings import ApplicationSettingsRepository
from server.repositories.database.schema import application_schema
from server.services.errors import BadRequestError, InternalServiceError

_GENERATION_FIELDS = (
    "max_new_tokens_deterministic",
    "max_new_tokens_concise",
    "max_new_tokens_detailed",
    "detailed_num_beams",
)

###############################################################################
class SettingsService:
    """Public projection and persistence boundary for application settings."""

    # -------------------------------------------------------------------------
    def __init__(
        self, repository: ApplicationSettingsRepository | None = None
    ) -> None:
        self.repository = repository or ApplicationSettingsRepository()

    # -------------------------------------------------------------------------
    def get_settings(self) -> ApplicationSettingsResponse:
        try:
            values = self.repository.get_settings()
        except (RuntimeError, ValueError) as exc:
            raise InternalServiceError(
                detail="Unable to load application settings.",
            ) from exc
        return self._response(values)

    # -------------------------------------------------------------------------
    def update_settings(
        self, patch: ApplicationSettingsPatch
    ) -> ApplicationSettingsResponse:
        if patch.inference is not None and application_schema().profile == "studio":
            raise BadRequestError(
                detail="XREPORT Studio has no inference settings.",
            )
        try:
            values = self.repository.update_public_settings(self._flat_patch(patch))
        except (RuntimeError, ValueError) as exc:
            raise InternalServiceError(
                detail="Unable to persist application settings.",
            ) from exc
        return self._response(values)

    # -------------------------------------------------------------------------
    def reset_settings(self) -> ApplicationSettingsResponse:
        try:
            values = self.repository.reset_public_settings(
                default_application_settings(application_schema().profile)
            )
        except (RuntimeError, ValueError) as exc:
            raise InternalServiceError(
                detail="Unable to reset application settings.",
            ) from exc
        return self._response(values)

    # -------------------------------------------------------------------------
    @staticmethod
    def _flat_patch(patch: ApplicationSettingsPatch) -> dict[str, object]:
        changes: dict[str, object] = {}
        if patch.global_settings is not None:
            if patch.global_settings.seed is not None:
                changes["global_seed"] = patch.global_settings.seed
        if patch.jobs is not None:
            if patch.jobs.polling_interval is not None:
                changes["job_polling_interval"] = patch.jobs.polling_interval
        if patch.inference is not None:
            if patch.inference.model_timeout is not None:
                changes["inference_model_timeout"] = patch.inference.model_timeout
            if patch.inference.model_idle_timeout is not None:
                changes["inference_model_idle_timeout"] = (
                    patch.inference.model_idle_timeout
                )
            for name in _GENERATION_FIELDS:
                value = getattr(patch.inference, name)
                if value is not None:
                    changes[f"inference_{name}"] = value
        changes.update(
            patch.model_dump(include={"theme_preference"}, exclude_none=True)
        )
        if not changes:  # pragma: no cover - guarded by the request model
            raise ValueError("At least one application setting must be provided")
        return changes

    # -------------------------------------------------------------------------
    @classmethod
    def _response(
        cls, values: ApplicationSettingsValues
    ) -> ApplicationSettingsResponse:
        return ApplicationSettingsResponse(
            values=cls._public_settings(values),
            defaults=cls._public_settings(default_application_settings(application_schema().profile)),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _public_settings(
        values: ApplicationSettingsValues,
    ) -> RuntimeApplicationSettings:
        inference = None
        if isinstance(values, InferenceSettingsValues):
            inference = RuntimeInferenceSettings(
                model_timeout=values.inference_model_timeout,
                model_idle_timeout=values.inference_model_idle_timeout,
                **{
                    name: getattr(values, f"inference_{name}")
                    for name in _GENERATION_FIELDS
                },
            )
        return RuntimeApplicationSettings(
            **{
                "global": RuntimeGlobalSettings(seed=values.global_seed),
                "jobs": RuntimeJobSettings(
                    polling_interval=values.job_polling_interval,
                ),
                "inference": inference,
                "theme_preference": values.theme_preference,
            }
        )

###############################################################################
@lru_cache(maxsize=1)
def get_settings_service() -> SettingsService:
    return SettingsService()


__all__ = ["SettingsService", "get_settings_service"]
