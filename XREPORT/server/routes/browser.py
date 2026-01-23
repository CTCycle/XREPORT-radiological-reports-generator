from __future__ import annotations

from typing import Any

import sqlalchemy
from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import MetaData, Table, select
from sqlalchemy.exc import SQLAlchemyError

from XREPORT.server.database.database import XREPORTDatabase, database
from XREPORT.server.schemas.browser import BrowseConfigResponse, TableDataResponse, TableInfo, TableListResponse
from XREPORT.server.utils.configurations.server import ServerSettings, server_settings
from XREPORT.server.utils.logger import logger


# -----------------------------------------------------------------------------
def build_display_name(table_name: str) -> str:
    return table_name.replace("_", " ").strip().title() or table_name


###############################################################################
class BrowserEndpoint:
    """Endpoint for database browsing operations."""

    def __init__(
        self,
        router: APIRouter,
        database: XREPORTDatabase,
        server_settings: ServerSettings,
    ) -> None:
        self.router = router
        self.database = database
        self.server_settings = server_settings

    # -----------------------------------------------------------------------------
    async def list_tables(self) -> TableListResponse:
        engine = self.database.backend.engine
        try:
            inspector = sqlalchemy.inspect(engine)
            table_names = sorted(inspector.get_table_names())
        except SQLAlchemyError as exc:
            logger.exception("Unable to list database tables.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Database connection error: {exc}",
            ) from exc
        tables = [
            TableInfo(table_name=table_name, display_name=build_display_name(table_name))
            for table_name in table_names
        ]
        return TableListResponse(tables=tables)

    # -----------------------------------------------------------------------------
    async def get_browse_config(self) -> BrowseConfigResponse:
        return BrowseConfigResponse(
            browse_batch_size=self.server_settings.database.browse_batch_size,
        )

    # -----------------------------------------------------------------------------
    async def get_table_data(
        self,
        table_name: str,
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> TableDataResponse:
        engine = self.database.backend.engine
        try:
            inspector = sqlalchemy.inspect(engine)
            table_names = set(inspector.get_table_names())
        except SQLAlchemyError as exc:
            logger.exception("Unable to inspect database tables.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Database connection error: {exc}",
            ) from exc
        if table_name not in table_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Table '{table_name}' not found.",
            )

        try:
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=engine)
            columns = [column.name for column in table.columns]
        except SQLAlchemyError as exc:
            logger.exception("Unable to reflect table %s.", table_name)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load table schema: {exc}",
            ) from exc

        pk_constraint = inspector.get_pk_constraint(table_name) or {}
        pk_columns = pk_constraint.get("constrained_columns") or []
        order_columns: list[Any] = []
        for column_name in pk_columns:
            if column_name in table.c:
                order_columns.append(table.c[column_name])
        if not order_columns and columns:
            order_columns.append(table.c[columns[0]])

        query = select(table).limit(limit).offset(offset)
        if order_columns:
            query = query.order_by(*order_columns)

        try:
            with engine.connect() as conn:
                rows = conn.execute(query).mappings().all()
        except SQLAlchemyError as exc:
            logger.exception("Unable to fetch data for table %s.", table_name)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load table data: {exc}",
            ) from exc
        data = [dict(row) for row in rows]

        try:
            row_count = self.database.count_rows(table_name)
        except Exception:  # noqa: BLE001
            row_count = 0

        return TableDataResponse(
            table_name=table_name,
            display_name=build_display_name(table_name),
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
            data=data,
        )

    # -----------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all browser-related routes."""
        self.router.add_api_route(
            "/tables",
            self.list_tables,
            methods=["GET"],
            response_model=TableListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/config",
            self.get_browse_config,
            methods=["GET"],
            response_model=BrowseConfigResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/data/{table_name}",
            self.get_table_data,
            methods=["GET"],
            response_model=TableDataResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
router = APIRouter(prefix="/data/browser", tags=["browser"])
browser_endpoint = BrowserEndpoint(
    router=router,
    database=database,
    server_settings=server_settings,
)
browser_endpoint.add_routes()
