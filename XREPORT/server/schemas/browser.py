from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


###############################################################################
class TableInfo(BaseModel):
    table_name: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)


###############################################################################
class TableListResponse(BaseModel):
    tables: list[TableInfo]


###############################################################################
class TableDataResponse(BaseModel):
    table_name: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    row_count: int = Field(..., ge=0)
    column_count: int = Field(..., ge=0)
    columns: list[str]
    data: list[dict[str, Any]]

