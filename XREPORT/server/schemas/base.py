from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from datetime import time
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from APP.server.utils.configurations import server_settings




###############################################################################
class GeneralModel(BaseModel):
    param_A: float = Field(..., ge=-90.0, le=90.0)
    param_B: float = Field(..., ge=-180.0, le=180.0)

    # -------------------------------------------------------------------------
    @field_validator("something", mode="before")
    @classmethod
    def validate_data(cls, value: str | None) -> str | None:
        if value is None:
            return None  




