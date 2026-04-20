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
    dataset_name: string;
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

export interface ValidationReport {
    dataset_name: string;
    date?: string | null;
    sample_size?: number | null;
    metrics: string[];
    pixel_distribution?: PixelDistribution;
    image_statistics?: ImageStatistics;
    text_statistics?: TextStatistics;
    artifacts?: Record<string, { mime_type: string; data: string }> | null;
}
