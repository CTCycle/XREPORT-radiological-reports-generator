from __future__ import annotations

from fastapi import APIRouter, status

from xreport_studio.domain.tokenizers import (
    TokenizerCatalogueResponse,
    TokenizerCheckRequest,
    TokenizerCheckResponse,
)
from xreport_studio.services.tokenizers import TokenizerService


###############################################################################
class TokenizerEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self, router: APIRouter, service: TokenizerService | None = None
    ) -> None:
        self.router = router
        self.service = service or TokenizerService()

    # -------------------------------------------------------------------------
    async def list_tokenizers(self) -> TokenizerCatalogueResponse:
        return await self.service.list()

    # -------------------------------------------------------------------------
    async def check_tokenizer(
        self, request: TokenizerCheckRequest
    ) -> TokenizerCheckResponse:
        return await self.service.check(request.tokenizer)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "",
            self.list_tokenizers,
            methods=["GET"],
            response_model=TokenizerCatalogueResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/check",
            self.check_tokenizer,
            methods=["POST"],
            response_model=TokenizerCheckResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/tokenizers", tags=["tokenizers"])
    TokenizerEndpoint(router=router).add_routes()
    return router


router = get_router()
