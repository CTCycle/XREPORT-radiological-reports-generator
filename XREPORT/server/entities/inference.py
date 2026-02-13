from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


###############################################################################
@dataclass(frozen=True)
class InferenceImage:
    filename: str
    content_type: str
    data: bytes
    size_bytes: int


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
