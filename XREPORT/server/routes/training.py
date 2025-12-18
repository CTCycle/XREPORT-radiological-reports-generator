from __future__ import annotations

import os
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from XREPORT.server.schemas.training import (
    BrowseResponse,
    DatasetUploadResponse,
    DirectoryItem,
    ImagePathRequest,
    ImagePathResponse,
    LoadDatasetRequest,
    LoadDatasetResponse,
)
from XREPORT.server.database.database import database
from XREPORT.server.utils.constants import VALID_IMAGE_EXTENSIONS
from XREPORT.server.utils.logger import logger


router = APIRouter(prefix="/training", tags=["training"])

# Temporary storage for uploaded dataset
temp_dataset_storage: dict[str, Any] = {}


# -----------------------------------------------------------------------------
def scan_image_folder(folder_path: str) -> list[str]:
    if not os.path.isdir(folder_path):
        return []
    
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in VALID_IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, file))
    
    return image_paths


###############################################################################
@router.post(
    "/images/validate",
    response_model=ImagePathResponse,
    status_code=status.HTTP_200_OK,
)
async def validate_image_path(request: ImagePathRequest) -> ImagePathResponse:
    folder_path = request.folder_path.strip()
    
    if not folder_path:
        return ImagePathResponse(
            valid=False,
            folder_path=folder_path,
            image_count=0,
            message="Folder path cannot be empty",
        )
    
    if not os.path.exists(folder_path):
        return ImagePathResponse(
            valid=False,
            folder_path=folder_path,
            image_count=0,
            message=f"Path does not exist: {folder_path}",
        )
    
    if not os.path.isdir(folder_path):
        return ImagePathResponse(
            valid=False,
            folder_path=folder_path,
            image_count=0,
            message=f"Path is not a directory: {folder_path}",
        )
    
    image_paths = scan_image_folder(folder_path)
    image_count = len(image_paths)
    
    if image_count == 0:
        return ImagePathResponse(
            valid=False,
            folder_path=folder_path,
            image_count=0,
            message=f"No valid images found in: {folder_path}",
        )
    
    logger.info(f"Validated image folder: {folder_path} with {image_count} images")
    
    return ImagePathResponse(
        valid=True,
        folder_path=folder_path,
        image_count=image_count,
        message=f"Found {image_count} valid images",
    )


###############################################################################
@router.post(
    "/dataset/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
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
    
    try:
        contents = await file.read()
        
        if ext == ".csv":
            import io
            # Use sep=None to auto-detect delimiter (handles both comma and semicolon)
            df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python')
        else:  # .xlsx
            import io
            df = pd.read_excel(io.BytesIO(contents))
        
        # Store in temporary storage with unique key
        storage_key = f"dataset_{filename}"
        temp_dataset_storage[storage_key] = {
            "dataframe": df,
            "filename": filename,
        }
        
        logger.info(f"Dataset uploaded: {filename} with {len(df)} rows, {len(df.columns)} columns")
        
        return DatasetUploadResponse(
            success=True,
            filename=filename,
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


###############################################################################
@router.post(
    "/dataset/load",
    response_model=LoadDatasetResponse,
    status_code=status.HTTP_200_OK,
)
async def load_dataset(request: LoadDatasetRequest) -> LoadDatasetResponse:
    folder_path = request.image_folder_path.strip()
    sample_size = request.sample_size
    seed = request.seed
    
    # Validate folder path
    if not os.path.isdir(folder_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image folder path: {folder_path}",
        )
    
    # Scan for images
    image_paths = scan_image_folder(folder_path)
    if not image_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No valid images found in: {folder_path}",
        )
    
    # Check if we have an uploaded dataset
    if not temp_dataset_storage:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No dataset uploaded. Please upload a CSV/XLSX file first.",
        )
    
    # Get the most recently uploaded dataset
    latest_key = list(temp_dataset_storage.keys())[-1]
    dataset_info = temp_dataset_storage[latest_key]
    df: pd.DataFrame = dataset_info["dataframe"].copy()
    
    # Apply sample size if needed
    if sample_size < 1.0:
        df = df.sample(frac=sample_size, random_state=seed)
    
    # Build image name to path mapping
    images_mapping = {}
    for path in image_paths:
        basename = os.path.basename(path)
        name_no_ext = os.path.splitext(basename)[0]
        images_mapping[name_no_ext] = path
    
    # Try to match images with dataset records
    # Look for 'image' column in dataset
    image_column = None
    for col in df.columns:
        if col.lower() in {"image", "filename", "file", "img", "image_name"}:
            image_column = col
            break
    
    if image_column is None:
        # Just return stats without matching
        logger.warning("No image column found in dataset for matching")
        return LoadDatasetResponse(
            success=True,
            total_images=len(image_paths),
            matched_records=0,
            unmatched_records=len(df),
            message=f"Loaded {len(image_paths)} images and {len(df)} records. No image column found for matching.",
        )
    
    # Match records to image paths
    df["_path"] = df[image_column].astype(str).str.split(".").str[0].map(images_mapping)
    matched = df.dropna(subset=["_path"])
    unmatched = len(df) - len(matched)
    
    logger.info(f"Dataset loaded: {len(matched)} matched, {unmatched} unmatched records")
    
    # Persist matched data to database (only 'image' and 'text' columns)
    if not matched.empty:
        try:
            # Prepare data for database - only keep 'image' and 'text' columns
            db_df = matched[[image_column, "text"]].copy()
            db_df = db_df.rename(columns={image_column: "image"})
            database.save_into_database(db_df, "RADIOGRAPHY_DATA")
            logger.info(f"Saved {len(db_df)} records to RADIOGRAPHY_DATA table")
        except Exception as e:
            logger.exception("Failed to save data to database")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save data to database: {str(e)}",
            ) from e
    
    # Clear temporary storage after loading
    temp_dataset_storage.clear()
    
    return LoadDatasetResponse(
        success=True,
        total_images=len(image_paths),
        matched_records=len(matched),
        unmatched_records=unmatched,
        message=f"Successfully loaded dataset with {len(matched)} matched records",
    )


###############################################################################
def get_windows_drives() -> list[str]:
    """Get list of available Windows drives."""
    drives = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            drives.append(drive)
    return drives


def count_images_in_folder(folder_path: str) -> int:
    """Quick count of image files in a folder (non-recursive for speed)."""
    try:
        count = 0
        for item in os.listdir(folder_path):
            ext = os.path.splitext(item)[1].lower()
            if ext in VALID_IMAGE_EXTENSIONS:
                count += 1
        return count
    except (PermissionError, OSError):
        return 0


###############################################################################
@router.get(
    "/browse",
    response_model=BrowseResponse,
    status_code=status.HTTP_200_OK,
)
async def browse_directory(
    path: str = Query("", description="Directory path to browse. Empty returns drives.")
) -> BrowseResponse:
    """Browse directories on the server filesystem."""
    
    # If no path provided or path is empty, return drives (Windows)
    if not path or path.strip() == "":
        drives = get_windows_drives()
        return BrowseResponse(
            current_path="",
            parent_path=None,
            items=[
                DirectoryItem(name=d, path=d, is_dir=True, image_count=0)
                for d in drives
            ],
            drives=drives,
        )
    
    path = path.strip()
    
    # Validate path exists
    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path not found: {path}",
        )
    
    if not os.path.isdir(path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a directory: {path}",
        )
    
    # Get parent path
    parent_path = os.path.dirname(path)
    if parent_path == path:  # At drive root
        parent_path = ""  # Return to drives list
    
    # List directory contents
    items: list[DirectoryItem] = []
    try:
        for item_name in sorted(os.listdir(path)):
            item_path = os.path.join(path, item_name)
            is_dir = os.path.isdir(item_path)
            
            # Only include directories (not files) for navigation
            if is_dir:
                image_count = count_images_in_folder(item_path)
                items.append(DirectoryItem(
                    name=item_name,
                    path=item_path,
                    is_dir=True,
                    image_count=image_count,
                ))
    except PermissionError:
        logger.warning(f"Permission denied accessing: {path}")
    except OSError as e:
        logger.warning(f"Error accessing {path}: {e}")
    
    # Also count images in current folder
    current_image_count = count_images_in_folder(path)
    
    return BrowseResponse(
        current_path=path,
        parent_path=parent_path if parent_path else None,
        items=items,
        drives=get_windows_drives(),
    )
