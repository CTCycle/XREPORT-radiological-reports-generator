"""XREPORT Inference application: routers, startup hooks and the ASGI app.

The data layout is selected by `XREPORT_APP=xreport`, which `entry.py` and the
launcher export before this module is imported."""

from __future__ import annotations

from fastapi import FastAPI

from xreport_shared.app_factory import create_application
from xreport_shared.common.constants import FASTAPI_TITLE

PROFILE = "xreport"


###############################################################################
def create_app() -> FastAPI:
    from xreport_inference.api.huggingface_access import router as huggingface_access_router
    from xreport_inference.api.inference import router as inference_router
    from xreport_inference.api.storage_settings import router as storage_settings_router
    from xreport_shared.api.checkpoint_exchange import get_import_router
    from xreport_shared.api.database_settings import router as database_settings_router
    from xreport_shared.api.jobs import router as jobs_router
    from xreport_shared.api.settings import router as settings_router
    from xreport_shared.api.system import router as system_router
    from xreport_shared.services.checkpoint_exchange import sweep_stale_staging

    return create_application(
        PROFILE,
        FASTAPI_TITLE,
        [
            inference_router,
            huggingface_access_router,
            storage_settings_router,
            get_import_router(),
            jobs_router,
            settings_router,
            database_settings_router,
            system_router,
        ],
        on_startup=sweep_stale_staging,
    )


app = create_app()
