from __future__ import annotations

import unicodedata

###############################################################################
def normalize_key(value: str) -> str:
    """Return the canonical key used for logical, case-insensitive identity."""

    return unicodedata.normalize("NFKC", str(value)).strip().casefold()
