from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from XREPORT.server.utils.variables import env_variables  # noqa: F401

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from XREPORT.server.utils.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from XREPORT.server.routes.browser import router as browser_router
from XREPORT.server.routes.upload import router as upload_router
from XREPORT.server.routes.preparation import router as preparation_router
from XREPORT.server.routes.training import router as training_router
from XREPORT.server.routes.validation import router as validation_router
from XREPORT.server.routes.inference import router as inference_router

###############################################################################
app = FastAPI(
    title=FASTAPI_TITLE,
    version=FASTAPI_VERSION,
    description=FASTAPI_DESCRIPTION,
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
