from __future__ import annotations

from pydantic import BaseModel


###############################################################################
class MemoryUsage(BaseModel):
    used_gb: float
    total_gb: float


###############################################################################
class CudaStatus(BaseModel):
    available: bool
    device_name: str | None = None


###############################################################################
class SystemStatusResponse(BaseModel):
    ram: MemoryUsage | None = None
    cuda: CudaStatus
    vram: MemoryUsage | None = None
