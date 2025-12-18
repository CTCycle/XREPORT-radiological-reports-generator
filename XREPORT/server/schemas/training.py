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


###############################################################################
class StartTrainingRequest(BaseModel):
    epochs: int = Field(10, ge=1, le=1000, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, le=256, description="Batch size for training")
    training_seed: int = Field(42, description="Random seed for training")
    num_encoders: int = Field(4, ge=1, le=12, description="Number of encoder layers")
    num_decoders: int = Field(4, ge=1, le=12, description="Number of decoder layers")
    embedding_dims: int = Field(256, ge=64, le=1024, description="Embedding dimensions")
    attention_heads: int = Field(8, ge=1, le=16, description="Number of attention heads")
    train_temp: float = Field(1.0, ge=0.1, le=2.0, description="Training temperature")
    freeze_img_encoder: bool = Field(False, description="Freeze image encoder weights")
    use_img_augmentation: bool = Field(False, description="Enable image augmentation")
    shuffle_with_buffer: bool = Field(True, description="Enable shuffle with buffer")
    shuffle_size: int = Field(1024, ge=1, description="Shuffle buffer size")
    save_checkpoints: bool = Field(True, description="Save checkpoints during training")
    use_tensorboard: bool = Field(False, description="Enable TensorBoard logging")
    use_mixed_precision: bool = Field(False, description="Enable mixed precision training")
    use_device_GPU: bool = Field(True, description="Use GPU for training")
    device_ID: int = Field(0, ge=0, description="GPU device ID")
    plot_training_metrics: bool = Field(True, description="Generate training plots")
    use_scheduler: bool = Field(False, description="Use learning rate scheduler")
    target_LR: float = Field(0.0001, ge=0.000001, le=0.1, description="Target learning rate")
    warmup_steps: int = Field(100, ge=0, description="Warmup steps for scheduler")


###############################################################################
class ResumeTrainingRequest(BaseModel):
    checkpoint: str = Field(..., description="Checkpoint name to resume from")
    additional_epochs: int = Field(10, ge=1, le=1000, description="Additional epochs to train")


###############################################################################
class CheckpointInfo(BaseModel):
    name: str
    epochs: int = 0
    loss: float = 0.0
    val_loss: float = 0.0


###############################################################################
class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointInfo]


###############################################################################
class TrainingStatusResponse(BaseModel):
    is_training: bool
    current_epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    val_accuracy: float = 0.0
    progress_percent: int = 0
    elapsed_seconds: int = 0


###############################################################################
class ProcessDatasetRequest(BaseModel):
    sample_size: float = Field(1.0, ge=0.01, le=1.0, description="Fraction of data to use")
    seed: int = Field(42, description="Random seed for sampling")
    validation_size: float = Field(0.2, ge=0.05, le=0.5, description="Fraction of data for validation")
    split_seed: int = Field(42, description="Random seed for train/val split")
    tokenizer: str = Field("bert-base-uncased", description="Hugging Face tokenizer ID")
    max_report_size: int = Field(200, ge=50, le=1000, description="Maximum token length for reports")


###############################################################################
class ProcessDatasetResponse(BaseModel):
    success: bool
    total_samples: int
    train_samples: int
    validation_samples: int
    vocabulary_size: int
    message: str


