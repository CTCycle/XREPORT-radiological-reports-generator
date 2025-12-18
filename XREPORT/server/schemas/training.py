from __future__ import annotations

from pydantic import BaseModel, Field


###############################################################################
class ImagePathRequest(BaseModel):
    folder_path: str = Field(..., description="Server-side folder path containing images")


###############################################################################
class ImagePathResponse(BaseModel):
    valid: bool
    folder_path: str
    image_count: int
    message: str


###############################################################################
class DatasetUploadResponse(BaseModel):
    success: bool
    filename: str
    row_count: int
    column_count: int
    columns: list[str]
    message: str


###############################################################################
class LoadDatasetRequest(BaseModel):
    image_folder_path: str = Field(..., description="Folder path containing X-ray images")
    sample_size: float = Field(1.0, ge=0.01, le=1.0, description="Fraction of data to use")
    seed: int = Field(42, description="Random seed for sampling")


###############################################################################
class LoadDatasetResponse(BaseModel):
    success: bool
    total_images: int
    matched_records: int
    unmatched_records: int
    message: str


###############################################################################
class DirectoryItem(BaseModel):
    name: str
    path: str
    is_dir: bool
    image_count: int = 0  # Only for directories, count of image files


###############################################################################
class BrowseResponse(BaseModel):
    current_path: str
    parent_path: str | None
    items: list[DirectoryItem]
    drives: list[str] = []  # Windows drives like C:, D:
