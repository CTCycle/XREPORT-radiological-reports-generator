from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from server.configurations.tokenizers import TOKENIZER_ID_PATTERN

###############################################################################
class ImagePathRequest(BaseModel):
    folder_path: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Server-side folder path containing images",
    )
    model_config = ConfigDict(extra="forbid")

###############################################################################
class ImagePathResponse(BaseModel):
    valid: bool
    folder_path: str
    image_count: int
    message: str

###############################################################################
class DatasetUploadResponse(BaseModel):
    success: bool
    upload_id: str
    filename: str
    dataset_name: str
    row_count: int
    column_count: int
    columns: list[str]
    message: str

###############################################################################
class LoadDatasetRequest(BaseModel):
    upload_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Explicit upload identifier returned by the dataset upload endpoint",
    )
    image_folder_path: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Folder path containing X-ray images",
    )
    sample_size: float = Field(
        ..., ge=0.01, le=1.0, description="Fraction of data to use"
    )
    confirm_unmatched: bool = Field(
        ...,
        description="Explicitly confirm importing matched rows when some images are unmatched",
    )
    model_config = ConfigDict(extra="forbid")

###############################################################################
class LoadDatasetResponse(BaseModel):
    success: bool
    total_images: int
    matched_records: int
    unmatched_records: int
    requires_confirmation: bool = False
    partial_import: bool = False
    message: str

###############################################################################
class FolderSelectionResponse(BaseModel):
    path: str | None = None

###############################################################################
class StartTrainingRequest(BaseModel):
    dataset_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Processed dataset name to use for training",
    )
    epochs: int = Field(..., ge=1, le=1000, description="Number of training epochs")
    batch_size: int = Field(..., ge=1, le=256, description="Batch size for training")
    gradient_accumulation_steps: int = Field(
        ...,
        ge=1,
        le=64,
        description="Batches whose gradients are accumulated before each optimizer step",
    )
    architecture_preset: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Key of the built-in or saved model architecture preset",
    )
    use_img_augmentation: bool = Field(..., description="Enable image augmentation")
    save_checkpoints: bool = Field(..., description="Save checkpoints during training")
    checkpoint_id: str | None = Field(
        None,
        min_length=1,
        max_length=128,
        description="Optional custom identifier for the checkpoint",
    )
    use_device_GPU: bool = Field(..., description="Use GPU for training")
    device_ID: int = Field(..., ge=0, description="GPU device ID")
    use_mixed_precision: bool = Field(
        ..., description="Enable mixed precision policy for GPU training"
    )
    dataloader_workers: int = Field(
        ..., ge=0, le=32, description="DataLoader worker processes"
    )
    prefetch_factor: int = Field(
        ..., ge=1, le=8, description="DataLoader prefetch factor"
    )
    pin_memory: bool = Field(..., description="Enable DataLoader pinned memory")
    persistent_workers: bool = Field(
        ..., description="Keep DataLoader workers persistent across epochs"
    )
    plot_training_metrics: bool = Field(..., description="Generate training plots")
    use_scheduler: bool = Field(..., description="Use learning rate scheduler")
    target_LR: float = Field(
        ..., ge=0.000001, le=0.1, description="Target learning rate"
    )
    warmup_steps: int = Field(..., ge=0, description="Warmup steps for scheduler")
    model_config = ConfigDict(extra="forbid")

###############################################################################
class ResumeTrainingRequest(BaseModel):
    checkpoint: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Checkpoint name to resume from",
    )
    additional_epochs: int = Field(
        ..., ge=1, le=1000, description="Additional epochs to train"
    )
    model_config = ConfigDict(extra="forbid")

###############################################################################
class CheckpointInfo(BaseModel):
    name: str
    epochs: int | None = None
    loss: float | None = None
    val_loss: float | None = None
    artifact_status: str = "ready"
    message: str | None = None

###############################################################################
class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointInfo]

###############################################################################
class ProcessDatasetRequest(BaseModel):
    dataset_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Dataset name to process",
    )
    custom_name: str | None = Field(
        None,
        min_length=1,
        max_length=128,
        description="Optional custom name for the processed dataset",
    )
    sample_size: float = Field(
        ..., ge=0.01, le=1.0, description="Fraction of data to use"
    )
    validation_size: float = Field(
        ..., ge=0.05, le=0.5, description="Fraction of data for validation"
    )
    tokenizer: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=TOKENIZER_ID_PATTERN,
        description="Hugging Face tokenizer ID",
    )
    max_report_size: int = Field(
        ..., ge=50, le=1000, description="Maximum token length for reports"
    )
    model_config = ConfigDict(extra="forbid")

###############################################################################
class ProcessDatasetResponse(BaseModel):
    success: bool
    total_samples: int
    train_samples: int
    validation_samples: int
    vocabulary_size: int
    message: str

###############################################################################
class DatasetStatusResponse(BaseModel):
    has_data: bool
    row_count: int
    message: str

###############################################################################
class DatasetInfo(BaseModel):
    name: str
    folder_path: str
    row_count: int
    has_validation_report: bool = False

###############################################################################
class DatasetNamesResponse(BaseModel):
    datasets: list[DatasetInfo]
    count: int

###############################################################################
class ProcessingMetadataResponse(BaseModel):
    dataset_name: str
    metadata: dict[str, Any]

###############################################################################
class CheckpointMetadataResponse(BaseModel):
    checkpoint: str
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    session: dict[str, Any]

###############################################################################
class DeleteResponse(BaseModel):
    success: bool
    message: str

###############################################################################
class ImageCountResponse(BaseModel):
    dataset_name: str
    count: int

###############################################################################
class ImageMetadataResponse(BaseModel):
    dataset_name: str
    index: int
    image_name: str
    caption: str
    valid_path: bool
    path: str
