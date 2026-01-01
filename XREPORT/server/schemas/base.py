from __future__ import annotations

from pydantic import BaseModel, Field


###############################################################################
class GeneralModel(BaseModel):
    param_A: float = Field(0.0)
    param_B: float = Field(0.0)

