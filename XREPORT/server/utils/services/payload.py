from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status


# HELPERS
###############################################################################
def sanitize_field(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None

#


##############################################################################
def sanitize_payload(
    *,
    this: str
) -> dict[str, Any]:    

    payload: dict[str, Any] = {"this": this}

    return payload
