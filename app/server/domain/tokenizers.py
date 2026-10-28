from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from server.configurations.tokenizers import TOKENIZER_ID_PATTERN

###############################################################################
class TokenizerOption(BaseModel):
    """One tokenizer offered by the catalogue."""

    tokenizer_id: str
    label: str
    group: str
    kind: str
    supports_training: bool
    cached: bool
    note: str | None = None

###############################################################################
class TokenizerCatalogueResponse(BaseModel):
    tokenizers: list[TokenizerOption]

###############################################################################
class TokenizerCheckRequest(BaseModel):
    tokenizer: str = Field(
        ..., min_length=1, max_length=128, pattern=TOKENIZER_ID_PATTERN
    )
    model_config = ConfigDict(extra="forbid")

###############################################################################
class TokenizerCheckResponse(BaseModel):
    """Result of loading a tokenizer and inspecting its special tokens."""

    tokenizer: str
    available: bool
    compatible: bool
    cached: bool
    message: str
    kind: str | None = None
    vocabulary_size: int | None = None
    revision: str | None = None
    start_token: str | None = None
    end_token: str | None = None
    pad_token: str | None = None
    start_token_id: int | None = None
    end_token_id: int | None = None
    pad_token_id: int | None = None
