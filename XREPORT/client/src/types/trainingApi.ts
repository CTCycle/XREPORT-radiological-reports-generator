export interface ImagePathResponse {
    valid: boolean;
    folder_path: string;
    image_count: number;
    message: string;
}

export interface DatasetUploadResponse {
    success: boolean;
    filename: string;
    dataset_name: string;
    row_count: number;
    column_count: number;
    columns: string[];
    message: string;
}

export interface LoadDatasetRequest {
    image_folder_path: string;
    sample_size?: number;
}

export interface LoadDatasetResponse {
    success: boolean;
    total_images: number;
    matched_records: number;
    unmatched_records: number;
    message: string;
}

export interface DirectoryItem {
    name: string;
    path: string;
    is_dir: boolean;
    image_count: number;
}

export interface BrowseResponse {
    current_path: string;
    parent_path: string | null;
    items: DirectoryItem[];
    drives: string[];
}

export interface DatasetStatusResponse {
    has_data: boolean;
    row_count: number;
    allow_server_browse: boolean;
    message: string;
}

export interface DatasetInfo {
    name: string;
    folder_path: string;
    row_count: number;
    has_validation_report: boolean;
}

export interface ProcessingMetadataResponse {
    dataset_name: string;
    metadata: Record<string, unknown>;
}

export interface CheckpointMetadataResponse {
    checkpoint: string;
    configuration: Record<string, unknown>;
    metadata: Record<string, unknown>;
    session: Record<string, unknown>;
}

export interface DeleteResponse {
    success: boolean;
    message: string;
}

export interface DatasetNamesResponse {
    datasets: DatasetInfo[];
    count: number;
}

export interface ImageCountResponse {
    dataset_name: string;
    count: number;
}

export interface ImageMetadataResponse {
    dataset_name: string;
    index: number;
    image_name: string;
    caption: string;
    valid_path: boolean;
    path: string;
}

export interface ProcessDatasetRequest {
    dataset_name: string;
    custom_name?: string;
    sample_size: number;
    validation_size: number;
    tokenizer: string;
    max_report_size: number;
}

export interface ProcessDatasetResponse {
    success: boolean;
    total_samples: number;
    train_samples: number;
    validation_samples: number;
    vocabulary_size: number;
    message: string;
}

export interface StartTrainingConfig {
    dataset_name: string;
    epochs: number;
    batch_size: number;
    num_encoders: number;
    num_decoders: number;
    embedding_dims: number;
    attention_heads: number;
    train_temp: number;
    freeze_img_encoder: boolean;
    use_img_augmentation: boolean;
    shuffle_with_buffer: boolean;
    shuffle_size: number;
    save_checkpoints: boolean;
    checkpoint_id?: string;
    use_device_GPU: boolean;
    device_ID: number;
    jit_compile: boolean;
    jit_backend: string;
    use_mixed_precision: boolean;
    dataloader_workers: number;
    prefetch_factor: number;
    pin_memory: boolean;
    persistent_workers: boolean;
    plot_training_metrics: boolean;
    use_scheduler: boolean;
    target_LR: number;
    warmup_steps: number;
}

export interface CheckpointInfo {
    name: string;
    epochs: number;
    loss: number;
    val_loss: number;
}

export interface CheckpointsResponse {
    checkpoints: CheckpointInfo[];
}

export interface TrainingStatusResponse {
    job_id?: string | null;
    is_training: boolean;
    current_epoch: number;
    total_epochs: number;
    loss: number;
    val_loss: number;
    accuracy: number;
    val_accuracy: number;
    progress_percent: number;
    elapsed_seconds: number;
    poll_interval?: number;
}
