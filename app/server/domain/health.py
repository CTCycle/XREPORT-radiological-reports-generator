from __future__ import annotations

from pydantic import BaseModel


###############################################################################
class HealthResponse(BaseModel):
    status: str
    application: str
    version: str
    runtime_mode: str
    runtime_variant: str | None = None
    runtime_port: int | None = None


###############################################################################
class ShutdownResponse(BaseModel):
    status: str
