from __future__ import annotations

import asyncio

from xreport_studio.configurations.tokenizers import TOKENIZER_CATALOGUE, catalogue_entry
from xreport_studio.domain.tokenizers import (
    TokenizerCatalogueResponse,
    TokenizerCheckResponse,
    TokenizerOption,
)
from xreport_shared.models.training.processing import (
    TokenizerCompatibilityError,
    TokenizerUnavailableError,
    cached_tokenizer_revision,
    is_cached_tokenizer,
    load_tokenizer,
    resolve_special_tokens,
)


###############################################################################
def list_tokenizers() -> TokenizerCatalogueResponse:
    return TokenizerCatalogueResponse(
        tokenizers=[
            TokenizerOption(
                tokenizer_id=item.tokenizer_id,
                label=item.label,
                group=item.group,
                kind=item.kind,
                supports_training=item.supports_training,
                cached=is_cached_tokenizer(item.tokenizer_id),
                note=item.note,
            )
            for item in TOKENIZER_CATALOGUE
        ]
    )


###############################################################################
def check_tokenizer(tokenizer_id: str) -> TokenizerCheckResponse:
    """Load tokenizer assets (no model weights) and verify decoder compatibility."""
    entry = catalogue_entry(tokenizer_id)
    cached = is_cached_tokenizer(tokenizer_id)
    try:
        tokenizer = load_tokenizer(tokenizer_id)
    except TokenizerUnavailableError as exc:
        return TokenizerCheckResponse(
            tokenizer=tokenizer_id,
            available=False,
            compatible=False,
            cached=cached,
            message=str(exc),
        )
    details = {
        "tokenizer": tokenizer_id,
        "available": True,
        "cached": cached or is_cached_tokenizer(tokenizer_id),
        "kind": entry.kind if entry is not None else None,
        "vocabulary_size": len(tokenizer),
        "revision": cached_tokenizer_revision(tokenizer_id),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
    }
    try:
        special = resolve_special_tokens(tokenizer)
    except TokenizerCompatibilityError as exc:
        return TokenizerCheckResponse(**details, compatible=False, message=str(exc))
    return TokenizerCheckResponse(
        **{
            **details,
            "pad_token": special.pad_token,
            "pad_token_id": special.pad_token_id,
        },
        compatible=True,
        message="Tokenizer is compatible with report generation",
        start_token=special.start_token,
        end_token=special.end_token,
        start_token_id=special.start_token_id,
        end_token_id=special.end_token_id,
    )


###############################################################################
class TokenizerService:

    # -------------------------------------------------------------------------
    async def list(self) -> TokenizerCatalogueResponse:
        return await asyncio.to_thread(list_tokenizers)

    # -------------------------------------------------------------------------
    async def check(self, tokenizer_id: str) -> TokenizerCheckResponse:
        return await asyncio.to_thread(check_tokenizer, tokenizer_id)
