from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


class JsonDataSupport:
    """Shared normalization for JSON payloads and UTC timestamps."""

    @staticmethod
    def parse_json(value: Any, default: Any = None) -> Any:
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str) and value.strip():
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return default
            return decoded if isinstance(decoded, (dict, list)) else default
        return default

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)
