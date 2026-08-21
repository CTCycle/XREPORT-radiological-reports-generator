from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from server.services.errors import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    InternalServiceError,
    NotFoundError,
)

from server.repositories.database import get_database
from server.domain.training import (
    BrowseResponse,
    DatasetInfo,
    DatasetNamesResponse,
    DatasetStatusResponse,
    DirectoryItem,
    ImagePathRequest,
    ImagePathResponse,
    LoadDatasetRequest,
    LoadDatasetResponse,
    ProcessingMetadataResponse,
    DeleteResponse,
    ProcessDatasetRequest,
    ImageCountResponse,
    ImageMetadataResponse,
)
from server.domain.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from server.common.constants import VALID_IMAGE_EXTENSIONS
from server.common.utils.logger import logger
from server.services.jobs import JobManager, get_job_manager
from server.configurations.startup import get_server_settings
from server.services.upload import UploadState, get_upload_state
from server.repositories.preparation import PreparationRepository
from server.repositories.serialization.dataset import DatasetRepository
from server.common.constants import (
    DATASET_RECORDS_TABLE,
)
from server.configurations import ServerSettings
from server.services.dataset_processing import DatasetProcessingService

DATASET_NAME_EMPTY_ERROR = "Dataset name cannot be empty"
LOCAL_FILESYSTEM_DISABLED_ERROR = (
    "Local filesystem endpoints are disabled by server configuration"
)

###############################################################################
def _iter_image_paths(folder_path: str) -> Iterator[Path]:
    directory_path = Path(folder_path)
    if not directory_path.is_dir():
        return

    for root, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if Path(filename).suffix.lower() in VALID_IMAGE_EXTENSIONS:
                yield Path(root) / filename

###############################################################################
def scan_image_folder(
    folder_path: str,
    required_stems: set[str] | None = None,
) -> list[str]:
    required = {stem.casefold() for stem in required_stems} if required_stems else None
    return [
        str(image_path)
        for image_path in _iter_image_paths(folder_path)
        if required is None or image_path.stem.casefold() in required
    ]

###############################################################################
def count_image_files(folder_path: str) -> int:
    return sum(1 for _ in _iter_image_paths(folder_path))

###############################################################################
def get_windows_drives() -> list[str]:
    """Get list of available Windows drives."""
    drives = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = Path(f"{letter}:\\")
        if drive.exists():
            drives.append(str(drive))
    return drives

###############################################################################
def count_images_in_folder(folder_path: str) -> int:
    """Quick count of image files in a folder (non-recursive for speed)."""
    try:
        with os.scandir(folder_path) as entries:
            return sum(
                1
                for entry in entries
                if entry.is_file()
                and Path(entry.name).suffix.lower() in VALID_IMAGE_EXTENSIONS
            )
    except OSError:
        return 0

###############################################################################
class PreparationService:
    """Endpoint for dataset preparation and browsing operations."""

    JOB_TYPE = "dataset_processing"

    # -------------------------------------------------------------------------
    def __init__(
        self,
        repository: PreparationRepository,
        dataset_repository: DatasetRepository,
        processing_service: DatasetProcessingService,
        job_manager: JobManager,
        upload_state: UploadState,
        server_settings: ServerSettings,
    ) -> None:
        self.repository = repository
        self.dataset_repository = dataset_repository
        self.processing_service = processing_service
        self.job_manager = job_manager
        self.upload_state = upload_state
        self.server_settings = server_settings
        self.allow_local_filesystem_access = (
            self.server_settings.features.allow_local_filesystem_access
        )

    # -------------------------------------------------------------------------
    def ensure_local_filesystem_access(self) -> None:
        if self.allow_local_filesystem_access:
            return
        raise ForbiddenError(
            detail=LOCAL_FILESYSTEM_DISABLED_ERROR,
        )

    # -------------------------------------------------------------------------
    def get_image_column_name(self, columns: list[str]) -> str | None:
        candidate_names = {"image", "filename", "file", "img", "image_name"}
        for column in columns:
            if column.lower() in candidate_names:
                return column
        return None

    # -------------------------------------------------------------------------
    def build_images_mapping(self, image_paths: list[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for path in image_paths:
            image_path = Path(path)
            mapping[image_path.stem] = str(image_path)
        return mapping

    # -------------------------------------------------------------------------
    def prepare_dataset_records_dataframe(
        self,
        matched: pd.DataFrame,
        image_column: str,
        dataset_name: str,
    ) -> pd.DataFrame:
        db_df = matched.loc[:, [image_column, "text", "_path"]].copy()
        db_df.rename(
            columns={
                image_column: "image_name",
                "text": "report_text",
                "_path": "image_path",
            },
            inplace=True,
        )
        db_df["dataset_name"] = dataset_name
        db_df["row_order"] = range(1, len(db_df) + 1)
        return db_df.loc[
            :,
            [
                "dataset_name",
                "image_name",
                "report_text",
                "image_path",
                "row_order",
            ],
        ].copy()

    # -------------------------------------------------------------------------
    def get_job_status_or_404(self, job_id: str) -> dict[str, Any]:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise NotFoundError(
                detail=f"Job not found: {job_id}",
            )
        return job_status

    # -------------------------------------------------------------------------
    def get_dataset_status(self) -> DatasetStatusResponse:
        """Check if dataset is available in the database for processing."""
        row_count = self.repository.get_dataset_status()
        return DatasetStatusResponse(
            has_data=row_count > 0,
            row_count=row_count,
            allow_server_browse=self.allow_local_filesystem_access,
            message=f"Found {row_count} records in {DATASET_RECORDS_TABLE}"
            if row_count > 0
            else f"No data found in {DATASET_RECORDS_TABLE} table",
        )

    # -------------------------------------------------------------------------
    def get_dataset_names(self) -> DatasetNamesResponse:
        """Get list of distinct datasets with metadata (folder path, row count)."""
        rows = self.repository.get_dataset_names()

        datasets = []
        for row in rows:
            sample_path = row.sample_path or ""
            # Extract folder path from the sample path
            folder_path = str(Path(sample_path).parent) if sample_path else ""
            datasets.append(
                DatasetInfo(
                    name=row.name,
                    folder_path=folder_path,
                    row_count=int(row.row_count or 0),
                    has_validation_report=bool(row.has_validation_report),
                )
            )

        return DatasetNamesResponse(
            datasets=datasets,
            count=len(datasets),
        )

    # -------------------------------------------------------------------------
    def get_processed_dataset_names(self) -> DatasetNamesResponse:
        """Get list of processed datasets available for training."""
        rows = self.repository.get_processed_dataset_names()

        datasets = []
        for row in rows:
            datasets.append(
                DatasetInfo(
                    name=row.name,
                    folder_path="processed",  # Placeholder indicating processed data
                    row_count=int(row.row_count or 0),
                    has_validation_report=bool(row.has_validation_report),
                )
            )

        return DatasetNamesResponse(
            datasets=datasets,
            count=len(datasets),
        )

    # -------------------------------------------------------------------------
    def get_processing_metadata(
        self, dataset_name: str
    ) -> ProcessingMetadataResponse:
        dataset_name = dataset_name.strip()
        if not dataset_name:
            raise BadRequestError(
                detail=DATASET_NAME_EMPTY_ERROR,
            )

        latest_metadata = self.repository.get_processing_metadata(dataset_name)
        if not isinstance(latest_metadata, dict) or not latest_metadata:
            raise NotFoundError(
                detail="No processing metadata found",
            )

        metadata = dict(latest_metadata)
        metadata_name = str(metadata.pop("name", "") or dataset_name)
        metadata.pop("dataset_id", None)
        metadata.pop("source_dataset_id", None)
        metadata.pop("processing_run_id", None)

        return ProcessingMetadataResponse(
            dataset_name=metadata_name,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    def delete_dataset(self, dataset_name: str) -> DeleteResponse:
        dataset_name = dataset_name.strip()
        if not dataset_name:
            raise BadRequestError(
                detail=DATASET_NAME_EMPTY_ERROR,
            )
        deleted_count = self.repository.delete_dataset(dataset_name)
        if deleted_count <= 0:
            raise NotFoundError(
                detail=f"Dataset not found: {dataset_name}",
            )

        return DeleteResponse(
            success=True,
            message=f"Deleted dataset {dataset_name}",
        )

    # -------------------------------------------------------------------------
    def validate_image_path(self, request: ImagePathRequest) -> ImagePathResponse:
        self.ensure_local_filesystem_access()
        folder_path = request.folder_path.strip()

        if not folder_path:
            return ImagePathResponse(
                valid=False,
                folder_path=folder_path,
                image_count=0,
                message="Folder path cannot be empty",
            )

        directory_path = Path(folder_path)
        if not directory_path.exists():
            return ImagePathResponse(
                valid=False,
                folder_path=folder_path,
                image_count=0,
                message=f"Path does not exist: {folder_path}",
            )

        if not directory_path.is_dir():
            return ImagePathResponse(
                valid=False,
                folder_path=folder_path,
                image_count=0,
                message=f"Path is not a directory: {folder_path}",
            )

        image_count = count_image_files(folder_path)

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

    # -------------------------------------------------------------------------
    def load_dataset(self, request: LoadDatasetRequest) -> LoadDatasetResponse:  # noqa: C901
        self.ensure_local_filesystem_access()
        folder_path = request.image_folder_path.strip()
        sample_size = request.sample_size
        seed = self.server_settings.global_settings.seed

        # Validate folder path
        directory_path = Path(folder_path)
        if not directory_path.is_dir():
            raise BadRequestError(
                detail=f"Invalid image folder path: {folder_path}",
            )

        # Check if we have an uploaded dataset
        if self.upload_state.is_empty():
            raise BadRequestError(
                detail="No dataset uploaded. Please upload a CSV/XLSX file first.",
            )

        # Get the most recently uploaded dataset
        latest_upload = self.upload_state.get_latest()
        if latest_upload is None:
            raise BadRequestError(
                detail="No dataset uploaded. Please upload a CSV/XLSX file first.",
            )

        _, dataset_info = latest_upload
        df: pd.DataFrame = dataset_info["dataframe"].copy()
        dataset_name: str = dataset_info["dataset_name"]

        # Apply sample size if needed
        if sample_size < 1.0:
            df = df.sample(frac=sample_size, random_state=seed)

        if "text" not in df.columns:
            raise BadRequestError(
                detail="Dataset is missing the required 'text' column.",
            )

        image_column = self.get_image_column_name(list(df.columns))
        total_images = count_image_files(folder_path)
        if total_images == 0:
            raise BadRequestError(
                detail=f"No valid images found in: {folder_path}",
            )

        if image_column is None:
            raise BadRequestError(
                detail=(
                    "Dataset is missing an image column. Use one of: "
                    "image, filename, file, img, image_name."
                ),
            )

        required_stems = {
            Path(str(value)).stem for value in df[image_column].tolist()
        }
        image_paths = scan_image_folder(folder_path, required_stems=required_stems)
        images_mapping = self.build_images_mapping(image_paths)

        # Match records to image paths
        df["_path"] = df[image_column].map(
            lambda value: images_mapping.get(Path(str(value)).stem)
        )
        matched = df.dropna(subset=["_path"])
        unmatched = len(df) - len(matched)

        logger.info(
            f"Dataset loaded: {len(matched)} matched, {unmatched} unmatched records"
        )

        if matched.empty:
            raise BadRequestError(
                detail=(
                    f"No dataset rows matched images in {folder_path}. "
                    f"{unmatched} record(s) remain unmatched."
                ),
            )

        if unmatched > 0 and not request.confirm_unmatched:
            return LoadDatasetResponse(
                success=False,
                total_images=total_images,
                matched_records=len(matched),
                unmatched_records=unmatched,
                requires_confirmation=True,
                partial_import=False,
                message=(
                    f"Found {len(matched)} matched and {unmatched} unmatched records. "
                    "Review the count and confirm to import the matched rows."
                ),
            )

        # Persist matched data to database with name and path.
        if not matched.empty:
            try:
                db_df = self.prepare_dataset_records_dataframe(
                    matched=matched,
                    image_column=image_column,
                    dataset_name=dataset_name,
                )
                self.dataset_repository.upsert_source_dataset(db_df)
                logger.info(
                    f"Upserted {len(db_df)} records to {DATASET_RECORDS_TABLE} table (dataset: {dataset_name})"
                )
            except Exception as e:
                logger.exception("Failed to save data to database")
                raise InternalServiceError(
                    detail=f"Failed to save data to database: {str(e)}",
                ) from e

        # Clear temporary storage after loading
        self.upload_state.clear()

        return LoadDatasetResponse(
            success=True,
            total_images=total_images,
            matched_records=len(matched),
            unmatched_records=unmatched,
            requires_confirmation=False,
            partial_import=unmatched > 0,
            message=(
                f"Successfully loaded dataset with {len(matched)} matched records "
                f"from {total_images} images"
                + (
                    f" after explicit confirmation; {unmatched} unmatched records were not imported."
                    if unmatched > 0
                    else ""
                )
            ),
        )

    # -------------------------------------------------------------------------
    def process_dataset(self, request: ProcessDatasetRequest) -> JobStartResponse:
        """Process the loaded dataset: sanitize text, tokenize, and split into train/val sets."""
        if self.job_manager.is_job_running("dataset_processing"):
            raise ConflictError(
                detail="Dataset processing is already in progress",
            )

        configuration = request.model_dump()
        configuration["seed"] = self.server_settings.global_settings.seed
        dataset_name = configuration.get("dataset_name", "").strip()
        if not dataset_name:
            raise BadRequestError(
                detail=DATASET_NAME_EMPTY_ERROR,
            )

        # Quick validation - check if source data exists
        dataset = self.dataset_repository.load_source_dataset(
            sample_size=1.0,
            seed=configuration["seed"],
            dataset_name=dataset_name,
        )
        if dataset.empty:
            raise NotFoundError(
                detail=f"Dataset not found: {dataset_name}",
            )

        # Start background job
        job_id = self.job_manager.start_job(
            job_type=self.JOB_TYPE,
            runner=self.processing_service.run,
            kwargs={
                "configuration": configuration,
            },
        )

        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise InternalServiceError(
                detail="Failed to initialize dataset processing job",
            )

        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message=f"Dataset processing job started for {dataset_name} ({len(dataset)} samples)",
            poll_interval=self.server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def get_preparation_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.get_job_status_or_404(job_id)
        return JobStatusResponse(**job_status)

    # -------------------------------------------------------------------------
    def cancel_preparation_job(self, job_id: str) -> JobCancelResponse:
        self.get_job_status_or_404(job_id)

        success = self.job_manager.cancel_job(job_id)

        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )

    # -------------------------------------------------------------------------
    def browse_directory(
        self,
        path: str = "",
    ) -> BrowseResponse:
        """Browse directories on the server filesystem."""
        self.ensure_local_filesystem_access()

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
        directory_path = Path(path)
        if not directory_path.exists():
            raise NotFoundError(
                detail=f"Path not found: {path}",
            )

        if not directory_path.is_dir():
            raise NotFoundError(
                detail=f"Path is not a directory: {path}",
            )

        # Get parent path
        parent = directory_path.parent
        parent_path = str(parent)
        if parent == directory_path:  # At drive root
            parent_path = ""  # Return to drives list

        # List directory contents
        items: list[DirectoryItem] = []
        try:
            for item_path in sorted(directory_path.iterdir(), key=lambda item: item.name):
                is_dir = item_path.is_dir()

                # Only include directories (not files) for navigation
                if is_dir:
                    image_count = count_images_in_folder(str(item_path))
                    items.append(
                        DirectoryItem(
                            name=item_path.name,
                            path=str(item_path),
                            is_dir=True,
                            image_count=image_count,
                        )
                    )
        except PermissionError as exc:
            logger.warning("Permission denied accessing %s: %s", path, exc)

        return BrowseResponse(
            current_path=path,
            parent_path=parent_path if parent_path else None,
            items=items,
            drives=get_windows_drives(),
        )

    # -------------------------------------------------------------------------
    def get_dataset_image_count(self, dataset_name: str) -> ImageCountResponse:
        """Get total number of images in a dataset."""
        dataset_name = dataset_name.strip()
        dataset = self.repository.get_dataset_by_name(dataset_name)
        count = (
            self.repository.count_records(dataset.dataset_id)
            if dataset is not None
            else 0
        )

        return ImageCountResponse(dataset_name=dataset_name, count=count)

    # -------------------------------------------------------------------------
    def get_dataset_image_metadata(
        self, dataset_name: str, index: int
    ) -> ImageMetadataResponse:
        """Get metadata for a specific image by 1-based index (id)."""
        self.ensure_local_filesystem_access()
        dataset_name = dataset_name.strip()
        if index < 1:
            raise BadRequestError(
                detail="Image index must be >= 1",
            )
        dataset = self.repository.get_dataset_by_name(dataset_name)
        row = (
            self.repository.get_record_at_index(dataset.dataset_id, index - 1)
            if dataset is not None
            else None
        )

        if not row:
            raise NotFoundError(
                detail=f"Image index {index} not found in dataset {dataset_name}",
            )

        image_name = row[0]
        caption = row[1] or ""
        path = row[2] or ""

        # Check if file exists
        valid_path = Path(path).exists() if path else False

        return ImageMetadataResponse(
            dataset_name=dataset_name,
            index=index,
            image_name=image_name,
            caption=caption,
            valid_path=valid_path,
            path=path,
        )

    # -------------------------------------------------------------------------
    def get_dataset_image_content(self, dataset_name: str, index: int) -> str:
        """Get absolute image file path for endpoint response handling."""
        self.ensure_local_filesystem_access()
        dataset_name = dataset_name.strip()
        if index < 1:
            raise BadRequestError(
                detail="Image index must be >= 1",
            )
        dataset = self.repository.get_dataset_by_name(dataset_name)
        row = (
            self.repository.get_record_at_index(dataset.dataset_id, index - 1)
            if dataset is not None
            else None
        )

        if not row:
            raise NotFoundError(
                detail=f"Image index {index} not found in dataset {dataset_name}",
            )

        path = row[2]

        if not path or not Path(path).exists():
            # Elegant warning message as requested
            raise NotFoundError(
                detail=f"Source file not found at {path}",
            )

        return str(Path(path).resolve())

###############################################################################
@lru_cache(maxsize=1)
def get_preparation_service() -> PreparationService:
    database = get_database()
    dataset_repository = DatasetRepository(database)
    job_manager = get_job_manager()
    return PreparationService(
        repository=PreparationRepository(database),
        dataset_repository=dataset_repository,
        processing_service=DatasetProcessingService(
            repository=dataset_repository,
            job_manager=job_manager,
        ),
        job_manager=job_manager,
        upload_state=get_upload_state(),
        server_settings=get_server_settings(),
    )


