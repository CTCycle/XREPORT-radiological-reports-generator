from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from XREPORT.server.schemas.training import DatasetUploadResponse
from XREPORT.server.utils.logger import logger


router = APIRouter(prefix="/upload", tags=["upload"])

# Shared temporary storage for uploaded dataset
# This is imported by other route modules (training.py, preparation.py)
temp_dataset_storage: dict[str, Any] = {}


###############################################################################
@router.post(
    "/dataset",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
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
        temp_dataset_storage[storage_key] = {
            "dataframe": df,
            "filename": filename,
            "dataset_name": dataset_name,
        }
        
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
