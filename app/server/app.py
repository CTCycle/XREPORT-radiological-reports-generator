from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse

from server.api.inference import router as inference_router
from server.api.errors import register_service_error_handlers
from server.api.preparation import router as preparation_router
from server.api.training import router as training_router
from server.api.upload import router as upload_router
from server.api.validation import router as validation_router
from server.common.constants import (
    FASTAPI_API_PREFIX,
    FASTAPI_DESCRIPTION,
    FASTAPI_DOCS_ENDPOINT,
    FASTAPI_ROOT_ENDPOINT,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from server.common.desktop_security import (
    DesktopSecurityMiddleware,
    SHUTDOWN_PATH,
    install_shutdown_event,
    desktop_token,
)
from server.common.path import CLIENT_DIST_DIR, PACKAGED_MODE
from server.common.path import RUNTIME_VARIANT
from server.configurations import get_server_settings
from server.domain.health import HealthResponse, ShutdownResponse
from server.services.startup_validation import run_startup_validations
from server.common.utils.logger import logger

###############################################################################
def redirect_root_to_docs() -> RedirectResponse | FileResponse:
    if PACKAGED_MODE:
        return FileResponse(CLIENT_DIST_DIR / "index.html")
    return RedirectResponse(FASTAPI_DOCS_ENDPOINT)

###############################################################################
def health_check(request: Request) -> HealthResponse:
    settings = getattr(request.app.state, "server_settings", None)
    runtime_mode = settings.database.backend if settings is not None else "unknown"
    return HealthResponse(
        status="ok",
        application=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        runtime_mode=runtime_mode,
        runtime_variant=RUNTIME_VARIANT,
        runtime_port=request.url.port,
    )


###############################################################################
def request_shutdown(request: Request) -> ShutdownResponse:
    event = getattr(request.app.state, "desktop_shutdown_event", None)
    if event is not None:
        event.set()
    return ShutdownResponse(status="shutting_down")


###############################################################################
async def wait_for_desktop_shutdown(
    shutdown_event: asyncio.Event,
    desktop_server: object,
) -> None:
    await shutdown_event.wait()
    desktop_server.should_exit = True  # type: ignore[attr-defined]


###############################################################################
def serve_packaged_frontend(path: str) -> FileResponse:
    relative = Path(path)
    candidate = (CLIENT_DIST_DIR / relative).resolve()
    try:
        candidate.relative_to(CLIENT_DIST_DIR.resolve())
    except ValueError:
        candidate = CLIENT_DIST_DIR / "index.html"
    if not candidate.is_file():
        candidate = CLIENT_DIST_DIR / "index.html"
    return FileResponse(candidate)

###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    startup_started = getattr(application.state, "desktop_startup_started_at", None)
    settings = get_server_settings()
    if startup_started is not None:
        logger.info(
            "Packaged startup phase=settings_loaded elapsed_ms=%.0f",
            (perf_counter() - startup_started) * 1000,
        )

    run_startup_validations(settings)
    if startup_started is not None:
        logger.info(
            "Packaged startup phase=startup_validations_completed elapsed_ms=%.0f",
            (perf_counter() - startup_started) * 1000,
        )

    application.state.server_settings = settings

    ready_callback = getattr(application.state, "desktop_ready_callback", None)
    if ready_callback is not None:
        ready_callback()
        if startup_started is not None:
            logger.info(
                "Packaged startup phase=ready_contract_written elapsed_ms=%.0f",
                (perf_counter() - startup_started) * 1000,
            )
    shutdown_task: asyncio.Task[None] | None = None
    shutdown_event = getattr(application.state, "desktop_shutdown_event", None)
    desktop_server = getattr(application.state, "desktop_server", None)
    if shutdown_event is not None and desktop_server is not None:
        shutdown_task = asyncio.create_task(
            wait_for_desktop_shutdown(shutdown_event, desktop_server)
        )

    try:
        yield
    finally:
        if shutdown_task is not None:
            shutdown_task.cancel()
            try:
                await shutdown_task
            except asyncio.CancelledError:
                pass

###############################################################################
def create_app() -> FastAPI:
    application = FastAPI(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
        lifespan=app_lifespan,
    )
    if PACKAGED_MODE:
        desktop_token()
        application.add_middleware(DesktopSecurityMiddleware)
        install_shutdown_event(application)
    register_service_error_handlers(application)

    for router in (
        upload_router,
        preparation_router,
        training_router,
        validation_router,
        inference_router,
    ):
        application.include_router(router, prefix=FASTAPI_API_PREFIX)
    application.add_api_route(
        "/api/health",
        health_check,
        methods=["GET"],
        response_model=HealthResponse,
        include_in_schema=False,
    )

    application.add_api_route(
        FASTAPI_ROOT_ENDPOINT,
        redirect_root_to_docs,
        methods=["GET"],
        include_in_schema=False,
        response_model=None,
    )
    if PACKAGED_MODE:
        application.add_api_route(
            SHUTDOWN_PATH,
            request_shutdown,
            methods=["POST"],
            include_in_schema=False,
            response_model=ShutdownResponse,
        )
        application.add_api_route(
            "/{path:path}",
            serve_packaged_frontend,
            methods=["GET"],
            include_in_schema=False,
            response_model=None,
        )

    return application


app = create_app()
