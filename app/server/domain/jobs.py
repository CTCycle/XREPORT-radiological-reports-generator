from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

JobLifecycleStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


###############################################################################
class JobStartResponse(BaseModel):
    job_id: str
    job_type: str
    status: JobLifecycleStatus
    message: str
    poll_interval: float = 1.0


###############################################################################
class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: JobLifecycleStatus
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
