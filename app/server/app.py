from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import asyncio
from importlib import import_module
import hmac
import os
from pathlib import Path
import re
from time import perf_counter

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from sqlalchemy.exc import SQLAlchemyError

from server.api.errors import register_service_error_handlers
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
from server.common.database_switch import DatabaseRequestGate
from server.common.path import CLIENT_DIST_DIR, PACKAGED_MODE
from server.common.path import APP_PROFILE, RESOURCES_DIR, RUNTIME_VARIANT
from server.common.process_lock import DataFolderLock
from server.configurations import get_database_settings
from server.configurations.database_config import DatabaseConfigurationError
from server.configurations.runtime_environment import (
    get_runtime_environment_settings,
)
from server.domain.health import HealthResponse, ShutdownResponse
from server.services.startup_validation import run_startup_validations
from server.common.utils.logger import logger

STUDIO_TITLE = "XREPORT Studio"

###############################################################################
def application_title(profile: str) -> str:
    return STUDIO_TITLE if profile == "studio" else FASTAPI_TITLE

###############################################################################
def profile_routers(profile: str) -> list[APIRouter]:
    """Import and return only the routers that belong to the given profile.

    Modules of the other application are never imported, so they are absent from
    the OpenAPI schema and from frozen desktop builds.
    """
    from server.api.database_settings import router as database_settings_router
    from server.api.jobs import router as jobs_router
    from server.api.settings import router as settings_router

    shared = [jobs_router, settings_router, database_settings_router]
    if profile == "studio":
        # Resolved by name so static analysis of the XREPORT build never sees them.
        studio_modules = (
            "server.api.upload",
            "server.api.preparation",
            "server.api.training",
            "server.api.validation",
        )
        from server.api.checkpoint_exchange import get_export_router

        return [
            get_export_router(),
            *(import_module(name).router for name in studio_modules),
            *shared,
        ]
    from server.api.checkpoint_exchange import get_import_router
    from server.api.inference import router as inference_router

    return [inference_router, get_import_router(), *shared]

###############################################################################
def redirect_root_to_docs(*, serve_source_client: bool = False) -> RedirectResponse | FileResponse:
    if PACKAGED_MODE or serve_source_client:
        return FileResponse(CLIENT_DIST_DIR / "index.html")
    return RedirectResponse(FASTAPI_DOCS_ENDPOINT)

###############################################################################
def health_check(request: Request) -> HealthResponse:
    settings = getattr(request.app.state, "server_settings", None)
    runtime_mode = (
        settings.database.backend
        if settings is not None
        else getattr(request.app.state, "database_backend", "unknown")
    )
    return HealthResponse(
        status="ok"
        if getattr(request.app.state, "database_ready", False)
        else "database_recovery",
        application=application_title(
            getattr(request.app.state, "app_profile", APP_PROFILE)
        ),
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
def serve_source_frontend(path: str) -> FileResponse | JSONResponse:
    """Serve built Angular assets, with SPA fallback for extensionless routes."""
    normalized_path = path.replace("\\", "/")
    if normalized_path == "api" or normalized_path.startswith("api/"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    relative = Path(normalized_path)
    candidate = (CLIENT_DIST_DIR / relative).resolve()
    try:
        candidate.relative_to(CLIENT_DIST_DIR.resolve())
    except ValueError:
        return JSONResponse({"detail": "Static file not found."}, status_code=404)

    if candidate.is_file():
        return FileResponse(candidate)
    if relative.suffix:
        return JSONResponse({"detail": "Static file not found."}, status_code=404)
    return FileResponse(CLIENT_DIST_DIR / "index.html")


###############################################################################
def source_shutdown(
    request: Request,
    launch_token: str | None = Header(
        default=None, alias="X-XREPORT-Launch-Token"
    ),
) -> ShutdownResponse:
    expected_token = getattr(request.app.state, "source_launch_token", "")
    if not launch_token or not hmac.compare_digest(launch_token, expected_token):
        raise HTTPException(status_code=403, detail="Invalid launcher token")
    return request_shutdown(request)


###############################################################################
def install_runtime_shutdown(application: FastAPI, source_launch_token: str) -> None:
    if PACKAGED_MODE:
        desktop_token()
        application.add_middleware(DesktopSecurityMiddleware)
        install_shutdown_event(application)
    elif source_launch_token:
        application.state.source_launch_token = source_launch_token
        install_shutdown_event(application)


###############################################################################
def install_runtime_routes(
    application: FastAPI,
    source_launch_token: str,
    serve_source_client: bool,
) -> None:
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
        return
    if source_launch_token:
        application.add_api_route(
            SHUTDOWN_PATH,
            source_shutdown,
            methods=["POST"],
            include_in_schema=False,
            response_model=ShutdownResponse,
        )
    if serve_source_client:
        application.add_api_route(
            "/{path:path}",
            serve_source_frontend,
            methods=["GET"],
            include_in_schema=False,
            response_model=None,
        )

###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.database_ready = False
    application.state.database_error = None
    application.state.database_backend = "unknown"
    # Fail closed when another process already owns this application's data folder.
    data_lock = DataFolderLock(RESOURCES_DIR)
    data_lock.acquire()
    startup_started = getattr(application.state, "desktop_startup_started_at", None)
    if startup_started is not None:
        logger.info(
            "Packaged startup phase=lifespan_entered elapsed_ms=%.0f",
            (perf_counter() - startup_started) * 1000,
        )
    try:
        database_settings = get_database_settings()
        application.state.database_backend = database_settings.backend
        settings = run_startup_validations(database_settings)
        if getattr(application.state, "app_profile", APP_PROFILE) == "xreport":
            from server.services.checkpoint_exchange import sweep_stale_staging

            sweep_stale_staging()
        application.state.server_settings = settings
        application.state.database_ready = True
        if startup_started is not None:
            logger.info(
                "Packaged startup phase=startup_validations_completed elapsed_ms=%.0f",
                (perf_counter() - startup_started) * 1000,
            )
            logger.info(
                "Packaged startup phase=settings_loaded elapsed_ms=%.0f",
                (perf_counter() - startup_started) * 1000,
            )
    except (
        DatabaseConfigurationError,
        RuntimeError,
        ValueError,
        OSError,
        SQLAlchemyError,
    ) as exc:
        message = _sanitize_startup_error(str(exc))
        application.state.database_error = message
        application.state.server_settings = None
        logger.error("Database startup requires recovery: %s", message)

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
        data_lock.release()
        if shutdown_task is not None:
            shutdown_task.cancel()
            try:
                await shutdown_task
            except asyncio.CancelledError:
                pass

###############################################################################
def create_app(profile: str | None = None) -> FastAPI:
    profile = profile or APP_PROFILE
    runtime_environment = get_runtime_environment_settings()
    serve_source_client = (
        not PACKAGED_MODE
        and os.getenv("XREPORT_SERVE_CLIENT", "").strip().lower()
        in {"1", "true", "yes"}
    )
    source_launch_token = "" if PACKAGED_MODE else os.getenv("XREPORT_LAUNCH_TOKEN", "")
    if serve_source_client and not (CLIENT_DIST_DIR / "index.html").is_file():
        raise FileNotFoundError(
            f"Built Angular frontend was not found at {CLIENT_DIST_DIR / 'index.html'}"
        )
    application = FastAPI(
        title=application_title(profile),
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
        lifespan=app_lifespan,
    )
    application.state.app_profile = profile
    application.state.database_request_gate = DatabaseRequestGate()
    application.state.database_ready = False
    application.state.database_error = None
    application.state.database_backend = "unknown"

    @application.middleware("http")
    async def database_request_guard(request: Request, call_next):
        path = request.url.path
        database_control_prefix = "/api/settings/database"
        is_database_control = path == "/api/health" or (
            path == database_control_prefix
            or path.startswith(database_control_prefix + "/")
        )
        if not path.startswith("/api/") or is_database_control:
            return await call_next(request)
        if not getattr(request.app.state, "database_ready", False):
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Database setup is required. Use the Database section in Settings."
                },
            )
        gate = request.app.state.database_request_gate
        return await gate.run_request(lambda: call_next(request))

    install_runtime_shutdown(application, source_launch_token)
    register_service_error_handlers(application)

    for router in profile_routers(profile):
        application.include_router(router, prefix=FASTAPI_API_PREFIX)
    application.add_api_route(
        "/api/health",
        health_check,
        methods=["GET"],
        response_model=HealthResponse,
        include_in_schema=False,
    )

    def root_response() -> RedirectResponse | FileResponse:
        return redirect_root_to_docs(serve_source_client=serve_source_client)

    application.add_api_route(
        FASTAPI_ROOT_ENDPOINT,
        root_response,
        methods=["GET"],
        include_in_schema=False,
        response_model=None,
    )
    install_runtime_routes(application, source_launch_token, serve_source_client)

    if runtime_environment.cors.allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(runtime_environment.cors.allowed_origins),
            allow_credentials=runtime_environment.cors.allow_credentials,
            allow_methods=list(runtime_environment.cors.allow_methods),
            allow_headers=list(runtime_environment.cors.allow_headers),
        )

    return application


###############################################################################
def _sanitize_startup_error(message: str) -> str:
    first_line = message.splitlines()[0][:280] if message else "Database startup failed."
    first_line = re.sub(
        r"(?i)(password\s*[=:]\s*)[^\s,;]+", r"\1<redacted>", first_line
    )
    return re.sub(r"(?i)(://[^:/\s]+:)[^@\s]+@", r"\1<redacted>@", first_line)


app = create_app()
