from __future__ import annotations

from typing import Any

__all__ = ["DataRepositoryQueries", "TrainingRepositoryQueries"]


def __getattr__(name: str) -> Any:
    if name == "DataRepositoryQueries":
        from XREPORT.server.repositories.queries.data import DataRepositoryQueries

        return DataRepositoryQueries
    if name == "TrainingRepositoryQueries":
        from XREPORT.server.repositories.queries.training import TrainingRepositoryQueries

        return TrainingRepositoryQueries
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
