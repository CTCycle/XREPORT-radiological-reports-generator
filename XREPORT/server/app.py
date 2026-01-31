from __future__ import annotations


from XREPORT.server.utils.variables import env_variables  # noqa: F401

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from XREPORT.server.configurations import server_settings
from XREPORT.server.routes.browser import router as browser_router
from XREPORT.server.routes.upload import router as upload_router
from XREPORT.server.routes.preparation import router as preparation_router
from XREPORT.server.routes.training import router as training_router
from XREPORT.server.routes.validation import router as validation_router
from XREPORT.server.routes.inference import router as inference_router

###############################################################################
app = FastAPI(
    title=server_settings.fastapi.title,
    version=server_settings.fastapi.version,
    description=server_settings.fastapi.description,
)

app.include_router(browser_router)
app.include_router(upload_router)
app.include_router(preparation_router)
app.include_router(training_router)
app.include_router(validation_router)
app.include_router(inference_router)


@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")
