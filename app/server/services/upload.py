from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from server.services.errors import (
    BadRequestError,
    PayloadTooLargeError,
    ServiceError,
)

from server.common.utils.logger import logger
from server.common.utils.security import sanitize_dataset_name
from server.domain.training import DatasetUploadResponse


MAX_DATASET_UPLOAD_BYTES = 16 * 1024 * 1024

###############################################################################
class UploadState:
    """Encapsulates temporary dataset storage."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.storage: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    def store(self, key: str, data: dict[str, Any]) -> None:
        self.storage[key] = data

    # -------------------------------------------------------------------------
    def get_latest(self) -> tuple[str, dict[str, Any]] | None:
        if not self.storage:
            return None
        latest_key = list(self.storage.keys())[-1]
        return latest_key, self.storage[latest_key]

    # -------------------------------------------------------------------------
    def clear(self) -> None:
        self.storage.clear()

    # -------------------------------------------------------------------------
    def is_empty(self) -> bool:
        return len(self.storage) == 0

###############################################################################
class UploadService:

    # -------------------------------------------------------------------------
    def __init__(self, upload_state: UploadState) -> None:
        self.upload_state = upload_state

    # -------------------------------------------------------------------------
    def upload_dataset(self, filename: str, contents: bytes) -> DatasetUploadResponse:
        if not filename:
            raise BadRequestError(
                detail="No file provided",
            )

        filename = Path(filename.strip().replace("\\", "/")).name
        if not filename:
            raise BadRequestError(
                detail="Invalid file name",
            )
        ext = Path(filename).suffix.lower()

        if ext not in {".csv", ".xlsx"}:
            raise BadRequestError(
                detail=f"Invalid file type: {ext}. Only .csv and .xlsx files are allowed",
            )

        try:
            if len(contents) > MAX_DATASET_UPLOAD_BYTES:
                raise PayloadTooLargeError(
                    detail=(
                        "Dataset payload exceeds "
                        f"{MAX_DATASET_UPLOAD_BYTES // (1024 * 1024)} MB limit"
                    ),
                )

            raw_dataset_name = Path(filename).stem
            dataset_name = sanitize_dataset_name(raw_dataset_name)

            if ext == ".csv":
                df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")
            else:
                df = pd.read_excel(io.BytesIO(contents))

            storage_key = f"dataset_{dataset_name}"
            self.upload_state.store(
                storage_key,
                {
                    "dataframe": df,
                    "filename": filename,
                    "dataset_name": dataset_name,
                },
            )

            logger.info(
                f"Dataset uploaded: {filename} (name: {dataset_name}) with {len(df)} rows, {len(df.columns)} columns"
            )

            return DatasetUploadResponse(
                success=True,
                filename=filename,
                dataset_name=dataset_name,
                row_count=len(df),
                column_count=len(df.columns),
                columns=list(df.columns),
                message=f"Successfully parsed {filename}",
            )
        except ServiceError:
            raise
        except Exception as e:
            logger.exception(f"Failed to parse dataset file: {filename}")
            raise BadRequestError(
                detail=f"Failed to parse file: {str(e)}",
            ) from e

###############################################################################
@lru_cache(maxsize=1)
def get_upload_state() -> UploadState:
    return UploadState()
