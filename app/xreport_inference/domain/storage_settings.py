from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

StorageSource = Literal["environment", "env_file", "pointer", "default"]
StorageTargetState = Literal["new", "empty", "existing"]
StorageMode = Literal["copy", "use_existing"]


###############################################################################
class StorageFolder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    size_bytes: int


###############################################################################
class StorageStatus(BaseModel):
    """Active and pending storage root with the folders it owns."""

    model_config = ConfigDict(extra="forbid")

    active_root: str
    default_root: str
    pending_root: str | None = None
    source: StorageSource
    locked: bool
    restart_required: bool
    folders: list[StorageFolder]


###############################################################################
class StoragePathRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)


###############################################################################
class StorageApplyRequest(StoragePathRequest):
    mode: StorageMode = "copy"


###############################################################################
class StorageValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    resolved_path: str
    target_state: StorageTargetState | None = None
    required_bytes: int = 0
    free_bytes: int | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
