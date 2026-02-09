from __future__ import annotations

from pydantic import BaseModel


###############################################################################
class CheckpointInfo(BaseModel):
    name: str
    created: str | None = None


###############################################################################
class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointInfo]
    success: bool
    message: str


###############################################################################
class GenerationRequest(BaseModel):
    checkpoint: str
    generation_mode: str


###############################################################################
class GenerationResponse(BaseModel):
    success: bool
    message: str
    reports: dict[str, str] | None = None
