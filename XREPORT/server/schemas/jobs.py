from __future__ import annotations

from typing import Any

from pydantic import BaseModel


###############################################################################
class JobStartResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    message: str
    poll_interval: float = 1.0


###############################################################################
class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: float
    result: dict[str, Any] | None = None
    error: str | None = None


###############################################################################
class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]


###############################################################################
class JobCancelResponse(BaseModel):
    job_id: str
    success: bool
    message: str
