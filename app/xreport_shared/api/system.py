from __future__ import annotations

from fastapi import APIRouter, status

from xreport_shared.domain.system import SystemStatusResponse
from xreport_shared.services.system_status import SystemStatusService

###############################################################################
class SystemEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        router: APIRouter,
        service: SystemStatusService | None = None,
    ) -> None:
        self.router = router
        self.service = service or SystemStatusService()

    # -------------------------------------------------------------------------
    def get_status(self) -> SystemStatusResponse:
        return self.service.get_status()

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/status",
            self.get_status,
            methods=["GET"],
            response_model=SystemStatusResponse,
            status_code=status.HTTP_200_OK,
        )

###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/system", tags=["system"])
    SystemEndpoint(router=router).add_routes()
    return router

router = get_router()
