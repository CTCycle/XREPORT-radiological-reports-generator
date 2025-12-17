from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, time
from io import BytesIO
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from APP.server.schemas.base import GeneralModel

from APP.server.utils.configurations import server_settings
from APP.server.utils.logger import logger


router = APIRouter(prefix="/base", tags=["tags"])


###############################################################################
class Endpoint:
    def __init__(
        self,
        router: APIRouter,      
    ) -> None:
        self.router = router     

    # -------------------------------------------------------------------------
    def first_method(
        self,
        payload: GeneralModel | None,
        response_payload: dict[str, Any] | None,        
    ) -> dict[str, Any] | None:        
        if payload:
            # do something
            pass
        try:
            # do something
            pass           
       
        except (TypeError, ValueError):
            return None
    
        return response_payload 

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/base",
            self.first_method,
            methods=["POST"], # or get
            status_code=status.HTTP_200_OK,
        )

      

base_endpoint = Endpoint(router=router)
base_endpoint.add_routes()

