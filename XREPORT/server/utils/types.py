from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


# -----------------------------------------------------------------------------
def extract_positive_int(value: Any) -> int | None:
    candidate: int | None = None
    if isinstance(value, bool) or value is None:
        candidate = None
    elif isinstance(value, int):
        candidate = value
    elif isinstance(value, float):
        candidate = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            candidate = int(stripped)
        else:
            match = re.search(r"\d+", stripped)
            if match:
                candidate = int(match.group(0))
    else:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            candidate = None
    if candidate is None or candidate <= 0:
        return None
    return candidate


# -----------------------------------------------------------------------------
def coerce_positive_int(value: Any, default: int = 1) -> int:
    candidate = extract_positive_int(value)
    return candidate if candidate is not None else default


# -----------------------------------------------------------------------------
def coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


# -----------------------------------------------------------------------------
def coerce_int(
    value: Any, default: int, minimum: int | None = None, maximum: int | None = None
) -> int:
    candidate: int
    if isinstance(value, bool):
        candidate = int(value)
    else:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            candidate = default
    if minimum is not None and candidate < minimum:
        candidate = minimum
    if maximum is not None and candidate > maximum:
        candidate = maximum
    return candidate


# -----------------------------------------------------------------------------
def coerce_float(
    value: Any,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        candidate = default
    if minimum is not None and candidate < minimum:
        candidate = minimum
    if maximum is not None and candidate > maximum:
        candidate = maximum
    return candidate


# -----------------------------------------------------------------------------
def coerce_str(value: Any, default: str) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or default
    if value is None:
        return default
    return str(value).strip() or default


# -----------------------------------------------------------------------------
def coerce_str_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


# -----------------------------------------------------------------------------
def coerce_str_sequence(value: Any, default: Iterable[str]) -> tuple[str, ...]:
    items: list[str] = []
    if isinstance(value, str):
        candidates = [
            segment.strip() for segment in value.split(",") if segment.strip()
        ]
    elif isinstance(value, Iterable):
        candidates = []
        for item in value:
            if isinstance(item, str):
                trimmed = item.strip()
                if trimmed:
                    candidates.append(trimmed)
    else:
        candidates = list(default)
    seen: set[str] = set()
    for candidate in candidates or default:
        lowered = candidate.lower()
        if lowered not in seen:
            seen.add(lowered)
            items.append(lowered)
    return tuple(items)


# -----------------------------------------------------------------------------
def coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        candidates = list(value)
    elif value is None:
        candidates = []
    else:
        candidates = [value]
    normalized: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        text = (
            candidate.strip() if isinstance(candidate, str) else str(candidate).strip()
        )
        if text:
            normalized.append(text)
    return tuple(normalized)


__all__ = [
    "coerce_bool",
    "coerce_float",
    "coerce_int",
    "coerce_positive_int",
    "extract_positive_int",
    "coerce_str",
]
