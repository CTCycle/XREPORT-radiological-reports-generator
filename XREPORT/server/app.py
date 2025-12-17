from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from APP.server.utils.variables import env_variables
from APP.server.routes.endpoint import router as general_router
from APP.server.utils.configurations import server_settings

###############################################################################
app = FastAPI(
    title=server_settings.fastapi.title,
    version=server_settings.fastapi.version,
    description=server_settings.fastapi.description,
)

app.include_router(general_router)

@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")

