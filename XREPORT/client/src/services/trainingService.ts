// Training service API functions

export interface ImagePathResponse {
    valid: boolean;
    folder_path: string;
    image_count: number;
    message: string;
}

export interface DatasetUploadResponse {
    success: boolean;
    filename: string;
    row_count: number;
    column_count: number;
    columns: string[];
    message: string;
}

export interface LoadDatasetRequest {
    image_folder_path: string;
    sample_size?: number;
    seed?: number;
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

async function readJson<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
        throw new Error(`Unexpected response content-type: ${contentType}`);
    }
    return (await response.json()) as T;
}

/**
 * Validate an image folder path on the server
 */
export async function validateImagePath(
    folderPath: string
): Promise<{ result: ImagePathResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/preparation/images/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: folderPath }),
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<ImagePathResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Upload a CSV or XLSX dataset file
 */
export async function uploadDataset(
    file: File
): Promise<{ result: DatasetUploadResponse | null; error: string | null }> {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/preparation/dataset/upload', {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<DatasetUploadResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Load dataset by merging images folder with uploaded dataset
 */
export async function loadDataset(
    request: LoadDatasetRequest
): Promise<{ result: LoadDatasetResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/preparation/dataset/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_folder_path: request.image_folder_path,
                sample_size: request.sample_size ?? 1.0,
                seed: request.seed ?? 42,
            }),
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<LoadDatasetResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Browse server-side directories
 */
export async function browseDirectory(
    path: string = ''
): Promise<{ result: BrowseResponse | null; error: string | null }> {
    try {
        const url = path
            ? `/api/preparation/browse?path=${encodeURIComponent(path)}`
            : '/api/preparation/browse';
        const response = await fetch(url);
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<BrowseResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

// ============================================================================
// Dataset Processing API
// ============================================================================

export interface ProcessDatasetRequest {
    sample_size: number;
    seed: number;
    validation_size: number;
    split_seed: number;
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

/**
 * Process the loaded dataset (sanitize, tokenize, split)
 */
export async function processDataset(
    config: ProcessDatasetRequest
): Promise<{ result: ProcessDatasetResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/preparation/dataset/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<ProcessDatasetResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

// ============================================================================
// Training Pipeline API
// ============================================================================

export interface StartTrainingConfig {
    epochs: number;
    batch_size: number;
    training_seed: number;
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
    use_tensorboard: boolean;
    use_mixed_precision: boolean;
    use_device_GPU: boolean;
    device_ID: number;
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
    is_training: boolean;
    current_epoch: number;
    total_epochs: number;
    loss: number;
    val_loss: number;
    accuracy: number;
    val_accuracy: number;
    progress_percent: number;
    elapsed_seconds: number;
}

/**
 * Get list of available checkpoints
 */
export async function getCheckpoints(): Promise<{ result: CheckpointsResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/pipeline/checkpoints');
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<CheckpointsResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Get current training status
 */
export async function getTrainingStatus(): Promise<{ result: TrainingStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/pipeline/status');
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<TrainingStatusResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Start a new training session
 */
export async function startTraining(
    config: StartTrainingConfig
): Promise<{ result: TrainingStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/pipeline/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<TrainingStatusResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Resume training from a checkpoint
 */
export async function resumeTraining(
    checkpoint: string,
    additionalEpochs: number
): Promise<{ result: TrainingStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/pipeline/resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                checkpoint,
                additional_epochs: additionalEpochs,
            }),
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<TrainingStatusResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Stop current training session
 */
export async function stopTraining(): Promise<{ result: TrainingStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/pipeline/stop', {
            method: 'POST',
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<TrainingStatusResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

