from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, status

from server.domain.settings import (
    ApplicationSettingsPatch,
    ApplicationSettingsResponse,
)

if TYPE_CHECKING:
    from server.services.settings import SettingsService


class SettingsEndpoint:
    def __init__(
        self,
        router: APIRouter,
        service: SettingsService | None = None,
    ) -> None:
        self.router = router
        self._service = service

    # -------------------------------------------------------------------------
    @property
    def service(self) -> SettingsService:
        if self._service is None:
            from server.services.settings import get_settings_service

            self._service = get_settings_service()
        return self._service

    # -------------------------------------------------------------------------
    def get_settings(self) -> ApplicationSettingsResponse:
        return self.service.get_settings()

    # -------------------------------------------------------------------------
    def update_settings(
        self, request: ApplicationSettingsPatch
    ) -> ApplicationSettingsResponse:
        return self.service.update_settings(request)

    # -------------------------------------------------------------------------
    def reset_settings(self) -> ApplicationSettingsResponse:
        return self.service.reset_settings()

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "",
            self.get_settings,
            methods=["GET"],
            response_model=ApplicationSettingsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "",
            self.update_settings,
            methods=["PATCH"],
            response_model=ApplicationSettingsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/reset",
            self.reset_settings,
            methods=["POST"],
            response_model=ApplicationSettingsResponse,
            status_code=status.HTTP_200_OK,
        )


def get_router() -> APIRouter:
    router = APIRouter(prefix="/settings", tags=["settings"])
    SettingsEndpoint(router=router).add_routes()
    return router


router = get_router()
