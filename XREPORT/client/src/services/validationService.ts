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

async function readJson<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
        throw new Error(`Unexpected response content-type: ${contentType}`);
    }
    return (await response.json()) as T;
}

/**
 * Run validation analytics on the current dataset
 */
export async function runValidation(
    request: ValidationRequest
): Promise<{ result: ValidationResponse | null; error: string | null }> {
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
        const payload = await readJson<ValidationResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}
