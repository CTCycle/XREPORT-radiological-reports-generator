// Training service API functions
import { readBoolean, readNumber, readString } from '../common/parsers';
import { JobCancelResponse, JobStartResponse, JobStatusResponse } from '../types/jobs';
import {
    BrowseResponse,
    CheckpointMetadataResponse,
    CheckpointsResponse,
    DatasetNamesResponse,
    DatasetStatusResponse,
    DatasetUploadResponse,
    DeleteResponse,
    ImageCountResponse,
    ImageMetadataResponse,
    ImagePathResponse,
    LoadDatasetRequest,
    LoadDatasetResponse,
    ProcessDatasetRequest,
    ProcessDatasetResponse,
    ProcessingMetadataResponse,
    StartTrainingConfig,
    TrainingStatusResponse,
} from '../types/trainingApi';

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
 * Get list of distinct processed dataset names available for training
 */
export async function getProcessedDatasetNames(): Promise<{ result: DatasetNamesResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/preparation/dataset/processed/names');
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

export function parseProcessDatasetResponse(result: Record<string, unknown>): ProcessDatasetResponse {
    return {
        success: readBoolean(result.success) ?? true,
        total_samples: readNumber(result.total_samples) ?? 0,
        train_samples: readNumber(result.train_samples) ?? 0,
        validation_samples: readNumber(result.validation_samples) ?? 0,
        vocabulary_size: readNumber(result.vocabulary_size) ?? 0,
        message: readString(result.message) ?? 'Dataset processed successfully',
    };
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
