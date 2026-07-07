from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from server.api.inference import router as inference_router
from server.api.preparation import router as preparation_router
from server.api.training import router as training_router
from server.api.upload import router as upload_router
from server.api.validation import router as validation_router
from server.common.constants import (
    FASTAPI_API_PREFIX,
    FASTAPI_ASSETS_ENDPOINT,
    FASTAPI_DESCRIPTION,
    FASTAPI_DOCS_ENDPOINT,
    FASTAPI_ROOT_ENDPOINT,
    FASTAPI_SPA_FALLBACK_ENDPOINT,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from server.common.path import (
    CLIENT_ASSETS_DIR,
    CLIENT_DIST_DIR,
    CLIENT_INDEX_FILE_PATH,
)
from server.configurations import get_server_settings
from server.repositories.database.initializer import initialize_database
from server.services.startup_validation import run_startup_validations

###############################################################################
def _client_build_available() -> bool:
    return CLIENT_INDEX_FILE_PATH.is_file()

###############################################################################
def _resolve_client_file(full_path: str) -> Path | None:
    client_root = CLIENT_DIST_DIR.resolve()
    requested_path = (client_root / full_path).resolve()

    if not requested_path.is_relative_to(client_root):
        return None

    if requested_path.is_file():
        return requested_path

    return None

###############################################################################
def serve_client_root() -> FileResponse:
    return FileResponse(CLIENT_INDEX_FILE_PATH)

###############################################################################
def serve_client_path(full_path: str) -> FileResponse:
    client_file = _resolve_client_file(full_path)
    if client_file is not None:
        return FileResponse(client_file)
    return FileResponse(CLIENT_INDEX_FILE_PATH)

###############################################################################
def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse(FASTAPI_DOCS_ENDPOINT)

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

    for router in (
        upload_router,
        preparation_router,
        training_router,
        validation_router,
        inference_router,
    ):
        application.include_router(router, prefix=FASTAPI_API_PREFIX)

    if _client_build_available():
        if CLIENT_ASSETS_DIR.is_dir():
            application.mount(
                FASTAPI_ASSETS_ENDPOINT,
                StaticFiles(directory=CLIENT_ASSETS_DIR),
                name="spa-assets",
            )
        application.add_api_route(
            FASTAPI_ROOT_ENDPOINT,
            serve_client_root,
            methods=["GET"],
            include_in_schema=False,
        )
        application.add_api_route(
            FASTAPI_SPA_FALLBACK_ENDPOINT,
            serve_client_path,
            methods=["GET"],
            include_in_schema=False,
        )
    else:
        application.add_api_route(
            FASTAPI_ROOT_ENDPOINT,
            redirect_root_to_docs,
            methods=["GET"],
            include_in_schema=False,
        )

    return application


app = create_app()
