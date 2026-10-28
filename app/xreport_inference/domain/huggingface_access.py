from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


###############################################################################
class HuggingFaceAccessStatus(BaseModel):
    """Token presence only; the token value is never returned."""

    model_config = ConfigDict(extra="forbid", strict=True)

    configured: bool
    source: Literal["environment", "stored"] | None = None


###############################################################################
class HuggingFaceTokenUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    token: SecretStr = Field(min_length=1, max_length=512)
