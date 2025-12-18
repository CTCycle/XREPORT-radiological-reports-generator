from __future__ import annotations

# CRITICAL: Load environment variables BEFORE any torch/keras imports
from XREPORT.server.utils.variables import env_variables  # noqa: F401

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from XREPORT.server.utils.configurations import server_settings
from XREPORT.server.routes.browser import router as browser_router
from XREPORT.server.routes.training import router as training_router
from XREPORT.server.routes.pipeline import router as pipeline_router

###############################################################################
app = FastAPI(
    title=server_settings.fastapi.title,
    version=server_settings.fastapi.version,
    description=server_settings.fastapi.description,
)

app.include_router(browser_router)
app.include_router(training_router)
app.include_router(pipeline_router)


@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")
