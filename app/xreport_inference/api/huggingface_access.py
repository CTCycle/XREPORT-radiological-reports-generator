from __future__ import annotations

from fastapi import APIRouter, status

from xreport_inference.domain.huggingface_access import (
    HuggingFaceAccessStatus,
    HuggingFaceTokenUpdate,
)
from xreport_inference.services.huggingface_access import HuggingFaceAccessService


###############################################################################
class HuggingFaceAccessEndpoint:

    # -------------------------------------------------------------------------
    def __init__(self, service: HuggingFaceAccessService | None = None) -> None:
        self.service = service or HuggingFaceAccessService()

    # -------------------------------------------------------------------------
    def get_status(self) -> HuggingFaceAccessStatus:
        return self.service.get_status()

    # -------------------------------------------------------------------------
    def save_token(self, payload: HuggingFaceTokenUpdate) -> HuggingFaceAccessStatus:
        return self.service.save_token(payload.token.get_secret_value())

    # -------------------------------------------------------------------------
    def clear_token(self) -> HuggingFaceAccessStatus:
        return self.service.clear_token()

    # -------------------------------------------------------------------------
    def add_routes(self, router: APIRouter) -> None:
        router.add_api_route(
            "",
            self.get_status,
            methods=["GET"],
            response_model=HuggingFaceAccessStatus,
            status_code=status.HTTP_200_OK,
        )
        router.add_api_route(
            "",
            self.save_token,
            methods=["PUT"],
            response_model=HuggingFaceAccessStatus,
            status_code=status.HTTP_200_OK,
        )
        router.add_api_route(
            "",
            self.clear_token,
            methods=["DELETE"],
            response_model=HuggingFaceAccessStatus,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/settings/huggingface", tags=["huggingface-access"])
    HuggingFaceAccessEndpoint().add_routes(router)
    return router


router = get_router()

__all__ = ["HuggingFaceAccessEndpoint", "router"]
