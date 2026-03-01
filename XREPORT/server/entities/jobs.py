from __future__ import annotations

import threading
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from pydantic import BaseModel


###############################################################################
@dataclass
class JobState:
    job_id: str
    job_type: str
    status: str
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=monotonic)
    completed_at: float | None = None
    stop_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # -------------------------------------------------------------------------
    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    # -------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "job_type": self.job_type,
                "status": self.status,
                "progress": self.progress,
                "result": self.result,
                "error": self.error,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
            }


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
