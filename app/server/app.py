from __future__ import annotations

import os
import warnings

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from server.api.inference import router as inference_router
from server.api.preparation import router as preparation_router
from server.api.training import router as training_router
from server.api.upload import router as upload_router
from server.api.validation import router as validation_router
from server.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from server.configurations.environment import load_environment

warnings.filterwarnings("ignore", category=FutureWarning)


def tauri_mode_enabled() -> bool:
    value = os.getenv("XREPORT_TAURI_MODE", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_client_dist_path() -> str:
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_path, "client", "dist")


def packaged_client_available() -> bool:
    return tauri_mode_enabled() and os.path.isdir(get_client_dist_path())


def serve_spa_root() -> FileResponse:
    return FileResponse(os.path.join(get_client_dist_path(), "index.html"))


def serve_spa_entrypoint(full_path: str) -> FileResponse:
    client_dist_path = get_client_dist_path()
    requested_path = os.path.join(client_dist_path, full_path)
    if os.path.isfile(requested_path):
        return FileResponse(requested_path)
    return FileResponse(os.path.join(client_dist_path, "index.html"))


def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


def create_app() -> FastAPI:
    load_environment()

    application = FastAPI(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
    )

    for router in (
        upload_router,
        preparation_router,
        training_router,
        validation_router,
        inference_router,
    ):
        application.include_router(router, prefix="/api")

    if packaged_client_available():
        assets_path = os.path.join(get_client_dist_path(), "assets")
        if os.path.isdir(assets_path):
            application.mount(
                "/assets",
                StaticFiles(directory=assets_path),
                name="spa-assets",
            )
        application.add_api_route(
            "/",
            serve_spa_root,
            methods=["GET"],
            include_in_schema=False,
        )
        application.add_api_route(
            "/{full_path:path}",
            serve_spa_entrypoint,
            methods=["GET"],
            include_in_schema=False,
        )
    else:
        application.add_api_route("/", redirect_to_docs, methods=["GET"])

    return application


###############################################################################
app = create_app()

