// Validation service API functions

export interface PixelDistribution {
    bins: number[];
    counts: number[];
}

export interface ImageStatistics {
    count: number;
    mean_height: number;
    mean_width: number;
    mean_pixel_value: number;
    std_pixel_value: number;
    mean_noise_std: number;
    mean_noise_ratio: number;
}

export interface TextStatistics {
    count: number;
    total_words: number;
    unique_words: number;
    avg_words_per_report: number;
    min_words_per_report: number;
    max_words_per_report: number;
}

export interface ValidationRequest {
    metrics: string[];
    sample_size?: number;
    seed?: number;
}

export interface ValidationResponse {
    success: boolean;
    message: string;
    pixel_distribution?: PixelDistribution;
    image_statistics?: ImageStatistics;
    text_statistics?: TextStatistics;
}

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
 * Run validation analytics on the current dataset
 * Returns a job_id for polling status
 */
export async function runValidation(
    request: ValidationRequest
): Promise<{ result: JobStartResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/validation/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
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
 * Get validation job status
 */
export async function getValidationJobStatus(
    jobId: string
): Promise<{ result: JobStatusResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/validation/jobs/${jobId}`);
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
 * Cancel a validation job
 */
export async function cancelValidationJob(
    jobId: string
): Promise<{ result: JobCancelResponse | null; error: string | null }> {
    try {
        const response = await fetch(`/api/validation/jobs/${jobId}`, {
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
 * Poll validation job until it completes, fails, or is cancelled
 * Returns a cleanup function to stop polling
 */
export function pollValidationJobStatus(
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

        const { result, error } = await getValidationJobStatus(jobId);

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
