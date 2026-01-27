from __future__ import annotations

import os
from typing import Any

import pandas as pd
import sqlalchemy
from fastapi import APIRouter, HTTPException, Query, status

from XREPORT.server.database.database import XREPORTDatabase, database
from XREPORT.server.schemas.training import (
    BrowseResponse,
    DatasetInfo,
    DatasetNamesResponse,
    DatasetStatusResponse,
    DirectoryItem,
    ImagePathRequest,
    ImagePathResponse,
    LoadDatasetRequest,
    LoadDatasetResponse,
    ProcessDatasetRequest,
    ProcessDatasetResponse,
)
from XREPORT.server.schemas.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from XREPORT.server.utils.constants import VALID_IMAGE_EXTENSIONS
from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.services.jobs import JobManager, job_manager
from XREPORT.server.utils.configurations.server import ServerSettings, server_settings
from XREPORT.server.routes.upload import UploadState, upload_state
from XREPORT.server.utils.services.training.serializer import DataSerializer
from XREPORT.server.utils.services.training.processing import (
    TextSanitizer,
    TokenizerHandler,
    TrainValidationSplit,
)

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


# -----------------------------------------------------------------------------
def get_windows_drives() -> list[str]:
    """Get list of available Windows drives."""
    drives = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            drives.append(drive)
    return drives


# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
def run_process_dataset_job(
    configuration: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Blocking dataset processing function that runs in background thread."""
    # Use the global job_manager imported at top level
    jm = job_manager
    
    serializer = DataSerializer()
    
    # Load source dataset from RADIOGRAPHY_DATA
    dataset = serializer.load_source_dataset(
        sample_size=configuration["sample_size"],
        seed=configuration["seed"],
    )
    
    if dataset.empty:
        raise RuntimeError("No data found in RADIOGRAPHY_DATA table. Please load a dataset first.")
    
    logger.info(f"Processing dataset with {len(dataset)} samples")
    
    # Update progress
    jm.update_progress(job_id, 10.0)
    if jm.should_stop(job_id):
        return {}
    
    # Step 1: Sanitize text corpus
    sanitizer = TextSanitizer(configuration)
    processed_data = sanitizer.sanitize_text(dataset)
    logger.info("Text sanitization completed")
    
    jm.update_progress(job_id, 30.0)
    if jm.should_stop(job_id):
        return {}
    
    # Step 2: Tokenize text using Hugging Face tokenizer
    try:
        tokenization = TokenizerHandler(configuration)
        logger.info(f"Tokenizing text using {tokenization.tokenizer_id} tokenizer")
        processed_data = tokenization.tokenize_text_corpus(processed_data)
        vocabulary_size = tokenization.vocabulary_size
        logger.info(f"Vocabulary size: {vocabulary_size} tokens")
    except Exception as e:
        logger.exception("Failed to tokenize text")
        raise RuntimeError(f"Tokenization failed: {str(e)}") from e
    
    jm.update_progress(job_id, 60.0)
    if jm.should_stop(job_id):
        return {}
    
    # Step 3: Drop raw text column (keep only tokens)
    processed_data = processed_data.drop(columns=["text"])
    
    # Step 4: Split into train and validation sets
    splitter = TrainValidationSplit(configuration, processed_data)
    training_data = splitter.split_train_and_validation()
    
    train_samples = len(training_data[training_data["split"] == "train"])
    validation_samples = len(training_data[training_data["split"] == "validation"])
    
    logger.info(f"Split complete: {train_samples} train, {validation_samples} validation samples")
    
    jm.update_progress(job_id, 80.0)
    if jm.should_stop(job_id):
        return {}
    
    # Step 5: Save processed data and metadata to database
    try:
        serializer.save_training_data(configuration, training_data, vocabulary_size)
        logger.info("Preprocessed data saved to database")
    except RuntimeError as e:
        # Schema mismatch or other runtime errors - use the clean message directly
        logger.error(f"Database error: {e}")
        raise
    except Exception as e:
        logger.exception("Failed to save training data")
        raise RuntimeError(f"Failed to save training data: {str(e)}") from e
    
    jm.update_progress(job_id, 100.0)
    
    return {
        "total_samples": len(training_data),
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "vocabulary_size": vocabulary_size,
    }


###############################################################################
class PreparationEndpoint:
    """Endpoint for dataset preparation and browsing operations."""

    JOB_TYPE = "dataset_processing"

    def __init__(
        self,
        router: APIRouter,
        database: XREPORTDatabase,
        job_manager: JobManager,
        upload_state: UploadState,
        server_settings: ServerSettings,
    ) -> None:
        self.router = router
        self.database = database
        self.job_manager = job_manager
        self.upload_state = upload_state
        self.server_settings = server_settings

    # -----------------------------------------------------------------------------
    async def get_dataset_status(self) -> DatasetStatusResponse:
        """Check if dataset is available in the database for processing."""
        row_count = self.database.count_rows("RADIOGRAPHY_DATA")
        return DatasetStatusResponse(
            has_data=row_count > 0,
            row_count=row_count,
            message=f"Found {row_count} records in RADIOGRAPHY_DATA" if row_count > 0 else "No data found in RADIOGRAPHY_DATA table",
        )

    # -----------------------------------------------------------------------------
    async def get_dataset_names(self) -> DatasetNamesResponse:
        """Get list of distinct datasets with metadata (folder path, row count)."""
        with self.database.backend.engine.connect() as conn:
            # Get dataset name, folder path (dirname of first path), and row count per dataset
            result = conn.execute(
                sqlalchemy.text('''
                    SELECT 
                        dataset_name,
                        MIN(path) as sample_path,
                        COUNT(*) as row_count
                    FROM "RADIOGRAPHY_DATA"
                    GROUP BY dataset_name
                    ORDER BY dataset_name
                ''')
            )
            datasets = []
            for row in result.fetchall():
                name = row[0]
                sample_path = row[1] or ""
                row_count = row[2]
                # Extract folder path from the sample path
                folder_path = os.path.dirname(sample_path) if sample_path else ""
                datasets.append(DatasetInfo(
                    name=name,
                    folder_path=folder_path,
                    row_count=row_count,
                ))
        
        return DatasetNamesResponse(
            datasets=datasets,
            count=len(datasets),
        )

    # -----------------------------------------------------------------------------
    async def validate_image_path(self, request: ImagePathRequest) -> ImagePathResponse:
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

    # -----------------------------------------------------------------------------
    async def load_dataset(self, request: LoadDatasetRequest) -> LoadDatasetResponse:
        folder_path = request.image_folder_path.strip()
        sample_size = request.sample_size
        seed = self.server_settings.global_settings.seed
        
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
        if self.upload_state.is_empty():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No dataset uploaded. Please upload a CSV/XLSX file first.",
            )
        
        # Get the most recently uploaded dataset
        _, dataset_info = self.upload_state.get_latest()
        df: pd.DataFrame = dataset_info["dataframe"].copy()
        dataset_name: str = dataset_info["dataset_name"]
        
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
        
        # Persist matched data to database with dataset_name, id, and path
        if not matched.empty:
            try:
                # Prepare data for database - include dataset_name, id, and path
                db_df = matched[[image_column, "text", "_path"]].copy()
                db_df = db_df.rename(columns={image_column: "image", "_path": "path"})
                db_df["dataset_name"] = dataset_name
                db_df["id"] = range(1, len(db_df) + 1)  # Incremental ID per dataset
                # Reorder columns to match schema: dataset_name, id, image, text, path
                db_df = db_df[["dataset_name", "id", "image", "text", "path"]]
                self.database.upsert_into_database(db_df, "RADIOGRAPHY_DATA")
                logger.info(f"Upserted {len(db_df)} records to RADIOGRAPHY_DATA table (dataset: {dataset_name})")
            except Exception as e:
                logger.exception("Failed to save data to database")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to save data to database: {str(e)}",
                ) from e
        
        # Clear temporary storage after loading
        self.upload_state.clear()
        
        return LoadDatasetResponse(
            success=True,
            total_images=len(image_paths),
            matched_records=len(matched),
            unmatched_records=unmatched,
            message=f"Successfully loaded dataset with {len(matched)} matched records",
        )

    # -----------------------------------------------------------------------------
    async def process_dataset(self, request: ProcessDatasetRequest) -> JobStartResponse:
        """Process the loaded dataset: sanitize text, tokenize, and split into train/val sets."""
        if self.job_manager.is_job_running("dataset_processing"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset processing is already in progress",
            )
        
        configuration = request.model_dump()
        configuration["seed"] = self.server_settings.global_settings.seed
        
        # Quick validation - check if source data exists
        row_count = self.database.count_rows("RADIOGRAPHY_DATA")
        if row_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data found in RADIOGRAPHY_DATA table. Please load a dataset first.",
            )
        
        # Start background job
        job_id = self.job_manager.start_job(
            job_type="dataset_processing",
            runner=run_process_dataset_job,
            kwargs={
                "configuration": configuration,
            },
        )

        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize dataset processing job",
            )
        
        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message=f"Dataset processing job started for {row_count} samples",
        )

    # -----------------------------------------------------------------------------
    async def get_preparation_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return JobStatusResponse(**job_status)

    # -----------------------------------------------------------------------------
    async def cancel_preparation_job(self, job_id: str) -> JobCancelResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        
        success = self.job_manager.cancel_job(job_id)
        
        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )

    # -----------------------------------------------------------------------------
    async def browse_directory(
        self,
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
                status_code=status.HTTP_404_NOT_FOUND,
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

    # -----------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all preparation-related routes."""
        self.router.add_api_route(
            "/dataset/status",
            self.get_dataset_status,
            methods=["GET"],
            response_model=DatasetStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/names",
            self.get_dataset_names,
            methods=["GET"],
            response_model=DatasetNamesResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/images/validate",
            self.validate_image_path,
            methods=["POST"],
            response_model=ImagePathResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/load",
            self.load_dataset,
            methods=["POST"],
            response_model=LoadDatasetResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/process",
            self.process_dataset,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_preparation_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_preparation_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/browse",
            self.browse_directory,
            methods=["GET"],
            response_model=BrowseResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
router = APIRouter(prefix="/preparation", tags=["preparation"])
preparation_endpoint = PreparationEndpoint(
    router=router,
    database=database,
    job_manager=job_manager,
    upload_state=upload_state,
    server_settings=server_settings,
)
preparation_endpoint.add_routes()
