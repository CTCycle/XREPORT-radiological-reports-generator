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


export interface JobStartResponse {
    job_id: string;
    job_type: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    message: string;
}

export interface JobStatusResponse {
    job_id: string;
    job_type: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    result: Record<string, unknown> | null;
    error: string | null;
}

export interface JobCancelResponse {
    job_id: string;
    success: boolean;
    message: string;
}

async function readJson<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
        throw new Error(`Unexpected response content-type: ${contentType}`);
    }
    return (await response.json()) as T;
}

/**
 * Check if dataset is available in the database for processing
 */
export async function getDatasetStatus(): Promise<{ result: DatasetStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/preparation/dataset/status');
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<DatasetStatusResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Get list of distinct dataset names available in the database
 */
export async function getDatasetNames(): Promise<{ result: DatasetNamesResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/preparation/dataset/names');
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<DatasetNamesResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Get processing metadata for a dataset
 */
export async function getProcessingMetadata(
    datasetName: string
): Promise<{ result: ProcessingMetadataResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/preparation/dataset/metadata/${encodeURIComponent(datasetName)}`);
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<ProcessingMetadataResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Delete a dataset and related metadata
 */
export async function deleteDataset(
    datasetName: string
): Promise<{ result: DeleteResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/preparation/dataset/${encodeURIComponent(datasetName)}`, {
            method: 'DELETE',
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<DeleteResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
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

        const response = await fetch('/api/upload/dataset', {
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
    dataset_name: string;
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

/**
 * Process the loaded dataset (sanitize, tokenize, split)
 * Returns a job_id for polling status
 */
export async function processDataset(
    config: ProcessDatasetRequest
): Promise<{ result: JobStartResponse | null; error: string | null }> {
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
        const payload = await readJson<JobStartResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Get preparation job status
 */
export async function getPreparationJobStatus(
    jobId: string
): Promise<{ result: JobStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/preparation/jobs/${jobId}`);
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<JobStatusResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Cancel a preparation job
 */
export async function cancelPreparationJob(
    jobId: string
): Promise<{ result: JobCancelResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/preparation/jobs/${jobId}`, {
            method: 'DELETE',
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<JobCancelResponse>(response);
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
}

/**
 * Get list of available checkpoints
 */
export async function getCheckpoints(): Promise<{ result: CheckpointsResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/training/checkpoints');
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
 * Get checkpoint metadata without loading the model
 */
export async function getCheckpointMetadata(
    checkpoint: string
): Promise<{ result: CheckpointMetadataResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/training/checkpoints/${encodeURIComponent(checkpoint)}/metadata`);
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<CheckpointMetadataResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Delete a checkpoint
 */
export async function deleteCheckpoint(
    checkpoint: string
): Promise<{ result: DeleteResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/training/checkpoints/${encodeURIComponent(checkpoint)}`, {
            method: 'DELETE',
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<DeleteResponse>(response);
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
        const response = await fetch('/api/training/status');
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
 * Returns a job_id for polling status
 */
export async function startTraining(
    config: StartTrainingConfig
): Promise<{ result: JobStartResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<JobStartResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Resume training from a checkpoint
 * Returns a job_id for polling status
 */
export async function resumeTraining(
    checkpoint: string,
    additionalEpochs: number
): Promise<{ result: JobStartResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/training/resume', {
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
        const payload = await readJson<JobStartResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Get training job status
 */
export async function getTrainingJobStatus(
    jobId: string
): Promise<{ result: JobStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/training/jobs/${jobId}`);
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<JobStatusResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Cancel a training job
 */
export async function cancelTrainingJob(
    jobId: string
): Promise<{ result: JobCancelResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/training/jobs/${jobId}`, {
            method: 'DELETE',
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<JobCancelResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Stop current training session (legacy endpoint)
 */
export async function stopTraining(): Promise<{ result: TrainingStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/training/stop', {
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

// ============================================================================
// Image Viewer API
// ============================================================================

export async function getDatasetImageCount(
    datasetName: string
): Promise<{ result: ImageCountResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/preparation/dataset/${encodeURIComponent(datasetName)}/images/count`);
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<ImageCountResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

export async function getDatasetImageMetadata(
    datasetName: string,
    index: number
): Promise<{ result: ImageMetadataResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/preparation/dataset/${encodeURIComponent(datasetName)}/images/${index}`);
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<ImageMetadataResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

export function getDatasetImageContentUrl(datasetName: string, index: number): string {
    return `/api/preparation/dataset/${encodeURIComponent(datasetName)}/images/${index}/content`;
}

// ============================================================================
// Job Polling Helpers
// ============================================================================

/**
 * Poll a job until it completes, fails, or is cancelled
 * Returns a cleanup function to stop polling
 */
export function pollJobStatus(
    getStatusFn: (jobId: string) => Promise<{ result: JobStatusResponse | null; error: string | null }>,
    jobId: string,
    onUpdate: (status: JobStatusResponse) => void,
    onComplete: (status: JobStatusResponse) => void,
    onError: (error: string) => void,
    intervalMs: number = 2000
): { stop: () => void } {
    let stopped = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
        if (stopped) return;

        const { result, error } = await getStatusFn(jobId);

        if (stopped) return;

        if (error) {
            onError(error);
            return;
        }

        if (!result) {
            onError('No result returned');
            return;
        }

        onUpdate(result);

        if (result.status === 'completed' || result.status === 'failed' || result.status === 'cancelled') {
            onComplete(result);
            return;
        }

        // Schedule next poll
        timeoutId = setTimeout(poll, intervalMs);
    };

    // Start polling
    poll();

    return {
        stop: () => {
            stopped = true;
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
        },
    };
}
