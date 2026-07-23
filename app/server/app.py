from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

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
from server.configurations import get_server_settings
from server.domain.health import HealthResponse
from server.repositories.database.initializer import initialize_database
from server.services.startup_validation import run_startup_validations

###############################################################################
def redirect_root_to_docs() -> RedirectResponse:
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
    )

###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_server_settings()

    initialize_database(settings.database)
    run_startup_validations(settings)

    application.state.server_settings = settings

    yield

###############################################################################
def create_app() -> FastAPI:
    application = FastAPI(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
        lifespan=app_lifespan,
    )
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
    )

    return application


app = create_app()
