from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from XREPORT.server.schemas.training import DatasetUploadResponse
from XREPORT.server.utils.logger import logger


###############################################################################
class UploadState:
    """Encapsulates temporary dataset storage."""

    def __init__(self) -> None:
        self.storage: dict[str, Any] = {}

    # -----------------------------------------------------------------------------
    def store(self, key: str, data: dict[str, Any]) -> None:
        self.storage[key] = data

    # -----------------------------------------------------------------------------
    def get_latest(self) -> tuple[str, dict[str, Any]] | None:
        if not self.storage:
            return None
        latest_key = list(self.storage.keys())[-1]
        return latest_key, self.storage[latest_key]

    # -----------------------------------------------------------------------------
    def clear(self) -> None:
        self.storage.clear()

    # -----------------------------------------------------------------------------
    def is_empty(self) -> bool:
        return len(self.storage) == 0


upload_state = UploadState()


###############################################################################
class UploadEndpoint:
    """Endpoint for dataset upload operations."""

    def __init__(self, router: APIRouter, upload_state: UploadState) -> None:
        self.router = router
        self.upload_state = upload_state

    # -----------------------------------------------------------------------------
    async def upload_dataset(self, file: UploadFile = File(...)) -> DatasetUploadResponse:
        """Upload a CSV or Excel file containing dataset records.
        
        The dataset name is extracted from the filename (without extension).
        The uploaded data is stored in temporary storage for later processing.
        """
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided",
            )
        
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        
        if ext not in {".csv", ".xlsx"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {ext}. Only .csv and .xlsx files are allowed",
            )
        
        # Extract dataset name from filename (without extension)
        dataset_name = os.path.splitext(filename)[0]
        
        try:
            contents = await file.read()
            
            if ext == ".csv":
                # Use sep=None to auto-detect delimiter (handles both comma and semicolon)
                df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python')
            else:  # .xlsx
                df = pd.read_excel(io.BytesIO(contents))
            
            # Store in temporary storage with unique key
            storage_key = f"dataset_{filename}"
            self.upload_state.store(storage_key, {
                "dataframe": df,
                "filename": filename,
                "dataset_name": dataset_name,
            })
            
            logger.info(f"Dataset uploaded: {filename} (name: {dataset_name}) with {len(df)} rows, {len(df.columns)} columns")
            
            return DatasetUploadResponse(
                success=True,
                filename=filename,
                dataset_name=dataset_name,
                row_count=len(df),
                column_count=len(df.columns),
                columns=list(df.columns),
                message=f"Successfully parsed {filename}",
            )
            
        except Exception as e:
            logger.exception(f"Failed to parse dataset file: {filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse file: {str(e)}",
            ) from e

    # -----------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all upload-related routes."""
        self.router.add_api_route(
            "/dataset",
            self.upload_dataset,
            methods=["POST"],
            response_model=DatasetUploadResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
router = APIRouter(prefix="/upload", tags=["upload"])
upload_endpoint = UploadEndpoint(router=router, upload_state=upload_state)
upload_endpoint.add_routes()
