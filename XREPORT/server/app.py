from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from XREPORT.server.common.utils.variables import env_variables  # noqa: F401

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from XREPORT.server.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from XREPORT.server.routes.upload import router as upload_router
from XREPORT.server.routes.preparation import router as preparation_router
from XREPORT.server.routes.training import router as training_router
from XREPORT.server.routes.validation import router as validation_router
from XREPORT.server.routes.inference import router as inference_router


def tauri_mode_enabled() -> bool:
    value = os.getenv("XREPORT_TAURI_MODE", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_client_dist_path() -> str:
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_path, "client", "dist")


###############################################################################
app = FastAPI(
    title=FASTAPI_TITLE,
    version=FASTAPI_VERSION,
    description=FASTAPI_DESCRIPTION,
)

routers = [
    upload_router,
    preparation_router,
    training_router,
    validation_router,
    inference_router,
]

for router in routers:
    app.include_router(router)
    app.include_router(router, prefix="/api", include_in_schema=False)

if tauri_mode_enabled() and os.path.isdir(get_client_dist_path()):
    app.mount("/", StaticFiles(directory=get_client_dist_path(), html=True), name="spa")


@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")

