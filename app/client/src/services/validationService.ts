// Validation service API functions
import { asRecord, readBoolean, readNumber, readNumberArray, readString } from '../common/parsers';
import { JobCancelResponse, JobStartResponse, JobStatusResponse } from '../types/jobs';
import {
    ImageStatistics,
    PixelDistribution,
    TextStatistics,
    ValidationReport,
    ValidationRequest,
    ValidationResponse,
} from '../types/validationApi';

function parsePixelDistribution(value: unknown): PixelDistribution | undefined {
    const payload = asRecord(value);
    if (!payload) {
        return undefined;
    }
    const bins = readNumberArray(payload.bins);
    const counts = readNumberArray(payload.counts);
    if (!bins || !counts) {
        return undefined;
    }
    return { bins, counts };
}

function parseImageStatistics(value: unknown): ImageStatistics | undefined {
    const payload = asRecord(value);
    if (!payload) {
        return undefined;
    }
    const count = readNumber(payload.count);
    const mean_height = readNumber(payload.mean_height);
    const mean_width = readNumber(payload.mean_width);
    const mean_pixel_value = readNumber(payload.mean_pixel_value);
    const std_pixel_value = readNumber(payload.std_pixel_value);
    const mean_noise_std = readNumber(payload.mean_noise_std);
    const mean_noise_ratio = readNumber(payload.mean_noise_ratio);
    if (
        count === undefined ||
        mean_height === undefined ||
        mean_width === undefined ||
        mean_pixel_value === undefined ||
        std_pixel_value === undefined ||
        mean_noise_std === undefined ||
        mean_noise_ratio === undefined
    ) {
        return undefined;
    }
    return {
        count,
        mean_height,
        mean_width,
        mean_pixel_value,
        std_pixel_value,
        mean_noise_std,
        mean_noise_ratio,
    };
}

function parseTextStatistics(value: unknown): TextStatistics | undefined {
    const payload = asRecord(value);
    if (!payload) {
        return undefined;
    }
    const count = readNumber(payload.count);
    const total_words = readNumber(payload.total_words);
    const unique_words = readNumber(payload.unique_words);
    const avg_words_per_report = readNumber(payload.avg_words_per_report);
    const min_words_per_report = readNumber(payload.min_words_per_report);
    const max_words_per_report = readNumber(payload.max_words_per_report);
    if (
        count === undefined ||
        total_words === undefined ||
        unique_words === undefined ||
        avg_words_per_report === undefined ||
        min_words_per_report === undefined ||
        max_words_per_report === undefined
    ) {
        return undefined;
    }
    return {
        count,
        total_words,
        unique_words,
        avg_words_per_report,
        min_words_per_report,
        max_words_per_report,
    };
}

export function parseValidationResponse(result: Record<string, unknown>): ValidationResponse {
    return {
        success: readBoolean(result.success) ?? true,
        message: readString(result.message) ?? 'Validation completed',
        pixel_distribution: parsePixelDistribution(result.pixel_distribution),
        image_statistics: parseImageStatistics(result.image_statistics),
        text_statistics: parseTextStatistics(result.text_statistics),
    };
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
 * Get a persisted validation report for a dataset
 */
export async function getValidationReport(
    datasetName: string
): Promise<{ result: ValidationReport | null; error: string | null }> {
    try {
        const response = await fetch(`/api/validation/reports/${encodeURIComponent(datasetName)}`);
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<ValidationReport>(response);
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

