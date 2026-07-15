// Inference service API functions
import { JobCancelResponse, JobStartResponse, JobStatusResponse } from '../types/jobs';
import {
    CheckpointEvaluationReport,
    CheckpointEvaluationRequest,
    GenerationProfile,
    InferenceModelsResponse,
} from '../types/inferenceApi';

async function readJson<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
        throw new Error(`Unexpected response content-type: ${contentType}`);
    }
    return (await response.json()) as T;
}

/**
 * Get the curated local inference model catalog.
 */
export async function getInferenceModels(): Promise<{
    result: InferenceModelsResponse | null;
    error: string | null;
}> {
    try {
        const response = await fetch('/api/inference/models');
        if (!response.ok) {
            const body = await response.text();
            return {
                result: null,
                error: `${response.status} ${response.statusText}: ${body}`,
            };
        }
        const payload = await readJson<InferenceModelsResponse>(response);
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
    modelRef: string,
    generationProfile: GenerationProfile,
    clinicalContext: string,
): Promise<{ result: JobStartResponse | null; error: string | null }> {
    try {
        const formData = new FormData();
        formData.append('model_ref', modelRef);
        formData.append('generation_profile', generationProfile);
        formData.append('clinical_context', clinicalContext);

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

/**
 * Evaluate a checkpoint using selected metrics
 * Returns a job_id for polling status
 */
export async function evaluateCheckpoint(
    checkpoint: string,
    metrics: string[],
    numSamples: number = 10,
    metricConfigs?: Record<string, { data_fraction?: number; num_samples?: number }>,
    seed?: number
): Promise<{ result: JobStartResponse | null; error: string | null }> {
    try {
        const request: CheckpointEvaluationRequest = {
            checkpoint,
            metrics,
            num_samples: numSamples,
            metric_configs: metricConfigs,
            seed,
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

/**
 * Get a persisted checkpoint evaluation report
 */
export async function getCheckpointEvaluationReport(
    checkpoint: string
): Promise<{ result: CheckpointEvaluationReport | null; error: string | null }> {
    try {
        const response = await fetch(
            `/api/validation/checkpoint/reports/${encodeURIComponent(checkpoint)}`
        );
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<CheckpointEvaluationReport>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

