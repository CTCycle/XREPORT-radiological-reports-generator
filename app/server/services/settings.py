from __future__ import annotations

from functools import lru_cache

from server.configurations import DEFAULT_APPLICATION_SETTINGS
from server.configurations.settings import ApplicationSettingsValues
from server.domain.settings import (
    ApplicationSettingsPatch,
    ApplicationSettingsResponse,
    RuntimeApplicationSettings,
    RuntimeFeatureSettings,
    RuntimeGlobalSettings,
    RuntimeInferenceSettings,
    RuntimeJobSettings,
)
from server.repositories.application_settings import ApplicationSettingsRepository
from server.services.errors import InternalServiceError


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
            values = self.repository.reset_public_settings(DEFAULT_APPLICATION_SETTINGS)
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
        if patch.features is not None:
            if patch.features.allow_local_filesystem_access is not None:
                changes["allow_local_filesystem_access"] = (
                    patch.features.allow_local_filesystem_access
                )
        if patch.jobs is not None:
            if patch.jobs.polling_interval is not None:
                changes["job_polling_interval"] = patch.jobs.polling_interval
        if patch.inference is not None:
            if patch.inference.model_timeout is not None:
                changes["inference_model_timeout"] = patch.inference.model_timeout
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
            defaults=cls._public_settings(DEFAULT_APPLICATION_SETTINGS),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _public_settings(
        values: ApplicationSettingsValues,
    ) -> RuntimeApplicationSettings:
        return RuntimeApplicationSettings(
            **{
                "global": RuntimeGlobalSettings(seed=values.global_seed),
                "features": RuntimeFeatureSettings(
                    allow_local_filesystem_access=values.allow_local_filesystem_access,
                ),
                "jobs": RuntimeJobSettings(
                    polling_interval=values.job_polling_interval,
                ),
                "inference": RuntimeInferenceSettings(
                    model_timeout=values.inference_model_timeout,
                ),
            }
        )


###############################################################################
@lru_cache(maxsize=1)
def get_settings_service() -> SettingsService:
    return SettingsService()


__all__ = ["SettingsService", "get_settings_service"]
