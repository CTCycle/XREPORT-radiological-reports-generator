from __future__ import annotations

from functools import lru_cache

from server.configurations.management import ConfigurationManager
from server.configurations.settings import JsonServerSettings, ServerSettings
from server.configurations.startup import (
    get_configuration_manager,
)
from server.domain.settings import (
    ApplicationSettingsPatch,
    ApplicationSettingsResponse,
    RuntimeApplicationSettings,
    RuntimeFeatureSettings,
    RuntimeGlobalSettings,
    RuntimeInferenceSettings,
    RuntimeJobSettings,
)
from server.services.errors import InternalServiceError


class SettingsService:
    """Public projection and persistence boundary for runtime settings."""

    def __init__(self, manager: ConfigurationManager | None = None) -> None:
        self.manager = manager or get_configuration_manager()

    # -------------------------------------------------------------------------
    def get_settings(self) -> ApplicationSettingsResponse:
        return self._response(self.manager.get_all())

    # -------------------------------------------------------------------------
    def update_settings(
        self, patch: ApplicationSettingsPatch
    ) -> ApplicationSettingsResponse:
        try:
            settings = self.manager.update_application_settings(
                patch.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude_unset=True,
                )
            )
        except RuntimeError as exc:
            raise InternalServiceError(
                detail="Unable to persist application settings.",
            ) from exc
        return self._response(settings)

    # -------------------------------------------------------------------------
    def reset_settings(self) -> ApplicationSettingsResponse:
        defaults = JsonServerSettings()
        patch = {
            "global": {"seed": defaults.global_settings.seed},
            "features": {
                "allow_local_filesystem_access": (
                    defaults.features.allow_local_filesystem_access
                )
            },
            "jobs": {"polling_interval": defaults.jobs.polling_interval},
            "inference": {"model_timeout": defaults.inference.model_timeout},
        }
        try:
            settings = self.manager.update_application_settings(patch)
        except RuntimeError as exc:
            raise InternalServiceError(
                detail="Unable to reset application settings.",
            ) from exc
        return self._response(settings)

    # -------------------------------------------------------------------------
    @classmethod
    def _response(cls, settings: ServerSettings) -> ApplicationSettingsResponse:
        return ApplicationSettingsResponse(
            values=cls._public_settings(settings),
            defaults=cls._default_public_settings(),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _public_settings(settings: ServerSettings) -> RuntimeApplicationSettings:
        return RuntimeApplicationSettings(
            **{
                "global": RuntimeGlobalSettings(
                    seed=settings.global_settings.seed,
                ),
                "features": RuntimeFeatureSettings(
                    allow_local_filesystem_access=(
                        settings.features.allow_local_filesystem_access
                    ),
                ),
                "jobs": RuntimeJobSettings(
                    polling_interval=settings.jobs.polling_interval,
                ),
                "inference": RuntimeInferenceSettings(
                    model_timeout=settings.inference.model_timeout,
                ),
            }
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _default_public_settings() -> RuntimeApplicationSettings:
        defaults = JsonServerSettings()
        return RuntimeApplicationSettings(
            **{
                "global": RuntimeGlobalSettings(seed=defaults.global_settings.seed),
                "features": RuntimeFeatureSettings(
                    allow_local_filesystem_access=(
                        defaults.features.allow_local_filesystem_access
                    ),
                ),
                "jobs": RuntimeJobSettings(
                    polling_interval=defaults.jobs.polling_interval,
                ),
                "inference": RuntimeInferenceSettings(
                    model_timeout=defaults.inference.model_timeout,
                ),
            }
        )


@lru_cache(maxsize=1)
def get_settings_service() -> SettingsService:
    return SettingsService(get_configuration_manager())
