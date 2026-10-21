from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from server.configurations.database_config import (
    DatabaseConfiguration,
    DatabaseConfigurationError,
)
from server.domain.database_settings import (
    DatabaseConfigurationStatusResponse,
    DatabaseConnectionTestResponse,
    DatabaseSwitchResponse,
)
from server.services.database_settings import (
    DatabaseSettingsService,
    DatabaseSwitchBusyError,
    get_database_settings_service,
)
from server.services.jobs import get_job_manager


class DatabaseSettingsEndpoint:
    def __init__(self, service: DatabaseSettingsService | None = None) -> None:
        self._service = service

    @property
    def service(self) -> DatabaseSettingsService:
        return self._service or get_database_settings_service()

    def get_status(self, request: Request) -> DatabaseConfigurationStatusResponse:
        return self.service.get_status(
            database_ready=bool(getattr(request.app.state, "database_ready", False)),
            startup_error=getattr(request.app.state, "database_error", None),
        )

    def test_connection(
        self, candidate: DatabaseConfiguration
    ) -> DatabaseConnectionTestResponse:
        return self.service.test_connection(candidate)

    async def switch_database(
        self, candidate: DatabaseConfiguration, request: Request
    ) -> DatabaseSwitchResponse:
        if get_job_manager().is_job_running():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Wait for active background jobs to finish before switching databases.",
            )

        gate = request.app.state.database_request_gate
        try:
            await gate.pause_and_drain()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc

        try:
            if get_job_manager().is_job_running():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Wait for active background jobs to finish before switching databases.",
                )
            result = await run_in_threadpool(self.service.switch_database, candidate)
            request.app.state.server_settings = result.server_settings
            request.app.state.database_ready = True
            request.app.state.database_backend = result.server_settings.database.backend
            request.app.state.database_error = None
            return result.response
        except DatabaseSwitchBusyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        except DatabaseConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Database switch failed: "
                    + _sanitize_switch_error(exc)
                    + " The previous configuration remains active."
                ),
            ) from exc
        finally:
            await gate.resume()

    def add_routes(self, router: APIRouter) -> None:
        router.add_api_route(
            "",
            self.get_status,
            methods=["GET"],
            response_model=DatabaseConfigurationStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        router.add_api_route(
            "/test",
            self.test_connection,
            methods=["POST"],
            response_model=DatabaseConnectionTestResponse,
            status_code=status.HTTP_200_OK,
        )
        router.add_api_route(
            "/switch",
            self.switch_database,
            methods=["POST"],
            response_model=DatabaseSwitchResponse,
            status_code=status.HTTP_200_OK,
        )


def get_router() -> APIRouter:
    router = APIRouter(prefix="/settings/database", tags=["database-settings"])
    DatabaseSettingsEndpoint().add_routes(router)
    return router


router = get_router()


def _sanitize_switch_error(error: Exception) -> str:
    message = str(error).splitlines()[0][:240] or "the target database could not be activated."
    message = re.sub(r"(?i)(password\s*[=:]\s*)[^\s,;]+", r"\1<redacted>", message)
    return re.sub(r"(?i)(://[^:/\s]+:)[^@\s]+@", r"\1<redacted>@", message)

__all__ = ["DatabaseSettingsEndpoint", "router"]
