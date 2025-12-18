from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from XREPORT.server.utils.variables import env_variables
from XREPORT.server.routes.browser import router as browser_router
from XREPORT.server.routes.training import router as training_router
from XREPORT.server.utils.configurations import server_settings

###############################################################################
app = FastAPI(
    title=server_settings.fastapi.title,
    version=server_settings.fastapi.version,
    description=server_settings.fastapi.description,
)

app.include_router(browser_router)
app.include_router(training_router)

@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")

