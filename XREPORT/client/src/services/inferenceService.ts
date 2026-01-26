// Inference service API functions

export interface CheckpointInfo {
    name: string;
    created: string | null;
}

export interface CheckpointsResponse {
    checkpoints: CheckpointInfo[];
    success: boolean;
    message: string;
}

export interface GenerationResponse {
    success: boolean;
    message: string;
    reports: Record<string, string> | null;
}

export type GenerationMode = 'greedy_search' | 'beam_search';

// ============================================================================
// Job API Types
// ============================================================================

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
 * Get list of available checkpoints for inference
 */
export async function getInferenceCheckpoints(): Promise<{
    result: CheckpointsResponse | null;
    error: string | null;
}> {
    try {
        const response = await fetch('/api/inference/checkpoints');
        if (!response.ok) {
            const body = await response.text();
            return {
                result: null,
                error: `${response.status} ${response.statusText}: ${body}`,
            };
        }
        const payload = await readJson<CheckpointsResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Generate reports for uploaded images
 * Returns a job_id for polling status
 */
export async function generateReports(
    images: File[],
    checkpoint: string,
    generationMode: GenerationMode
): Promise<{ result: JobStartResponse | null; error: string | null }> {
    try {
        const formData = new FormData();
        formData.append('checkpoint', checkpoint);
        formData.append('generation_mode', generationMode);

        for (const image of images) {
            formData.append('images', image);
        }

        const response = await fetch('/api/inference/generate', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const body = await response.text();
            return {
                result: null,
                error: `${response.status} ${response.statusText}: ${body}`,
            };
        }

        const payload = await readJson<JobStartResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Get inference job status
 */
export async function getInferenceJobStatus(
    jobId: string
): Promise<{ result: JobStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/inference/jobs/${jobId}`);
        if (!response.ok) {
            const body = await response.text();
            return {
                result: null,
                error: `${response.status} ${response.statusText}: ${body}`,
            };
        }
        const payload = await readJson<JobStatusResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Cancel an inference job
 */
export async function cancelInferenceJob(
    jobId: string
): Promise<{ result: JobCancelResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/inference/jobs/${jobId}`, {
            method: 'DELETE',
        });
        if (!response.ok) {
            const body = await response.text();
            return {
                result: null,
                error: `${response.status} ${response.statusText}: ${body}`,
            };
        }
        const payload = await readJson<JobCancelResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

// ============================================================================
// Checkpoint Evaluation
// ============================================================================

export interface CheckpointEvaluationRequest {
    checkpoint: string;
    metrics: string[];
    num_samples: number;
}

export interface CheckpointEvaluationResults {
    loss?: number;
    accuracy?: number;
    bleu_score?: number;
}

export interface CheckpointEvaluationResponse {
    success: boolean;
    message: string;
    results?: CheckpointEvaluationResults;
}

/**
 * Evaluate a checkpoint using selected metrics
 * Returns a job_id for polling status
 */
export async function evaluateCheckpoint(
    checkpoint: string,
    metrics: string[],
    numSamples: number = 10
): Promise<{ result: JobStartResponse | null; error: string | null }> {
    try {
        const request: CheckpointEvaluationRequest = {
            checkpoint,
            metrics,
            num_samples: numSamples,
        };

        const response = await fetch('/api/validation/checkpoint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const body = await response.text();
            return {
                result: null,
                error: `${response.status} ${response.statusText}: ${body}`,
            };
        }

        const payload = await readJson<JobStartResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Get checkpoint evaluation job status (from validation endpoint)
 */
export async function getCheckpointEvaluationJobStatus(
    jobId: string
): Promise<{ result: JobStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/validation/jobs/${jobId}`);
        if (!response.ok) {
            const body = await response.text();
            return {
                result: null,
                error: `${response.status} ${response.statusText}: ${body}`,
            };
        }
        const payload = await readJson<JobStatusResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

// ============================================================================
// Job Polling Helper
// ============================================================================

/**
 * Poll inference job until it completes, fails, or is cancelled
 * Returns a cleanup function to stop polling
 */
export function pollInferenceJobStatus(
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

        const { result, error } = await getInferenceJobStatus(jobId);

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
