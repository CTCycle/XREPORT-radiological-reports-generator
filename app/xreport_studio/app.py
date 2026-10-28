"""XREPORT Studio application: routers and the ASGI app.

The data layout is selected by `XREPORT_APP=studio`, which `entry.py` and the
launcher export before this module is imported."""

from __future__ import annotations

from fastapi import FastAPI

from xreport_shared.app_factory import create_application

PROFILE = "studio"
TITLE = "XREPORT Studio"


###############################################################################
def create_app() -> FastAPI:
    from xreport_shared.api.checkpoint_exchange import get_export_router
    from xreport_shared.api.database_settings import router as database_settings_router
    from xreport_shared.api.jobs import router as jobs_router
    from xreport_shared.api.settings import router as settings_router
    from xreport_shared.api.system import router as system_router
    from xreport_studio.api.model_architecture import router as model_architecture_router
    from xreport_studio.api.preparation import router as preparation_router
    from xreport_studio.api.tokenizers import router as tokenizers_router
    from xreport_studio.api.training import router as training_router
    from xreport_studio.api.upload import router as upload_router
    from xreport_studio.api.validation import router as validation_router

    return create_application(
        PROFILE,
        TITLE,
        [
            get_export_router(),
            upload_router,
            preparation_router,
            training_router,
            model_architecture_router,
            validation_router,
            tokenizers_router,
            jobs_router,
            settings_router,
            database_settings_router,
            system_router,
        ],
    )


app = create_app()
