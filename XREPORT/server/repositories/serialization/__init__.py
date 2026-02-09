from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from XREPORT.server.repositories.serialization.data import DataSerializer
    from XREPORT.server.repositories.serialization.model import ModelSerializer

__all__ = ["DataSerializer", "ModelSerializer"]


def __getattr__(name: str) -> Any:
    if name == "DataSerializer":
        from XREPORT.server.repositories.serialization.data import DataSerializer

        return DataSerializer
    if name == "ModelSerializer":
        from XREPORT.server.repositories.serialization.model import ModelSerializer

        return ModelSerializer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
