export interface CheckpointInfo {
    name: string;
    created: string | null;
}

export interface CheckpointsResponse {
    checkpoints: CheckpointInfo[];
    success: boolean;
    message: string;
}

export type GenerationMode = 'greedy_search' | 'beam_search';

export interface CheckpointEvaluationRequest {
    checkpoint: string;
    metrics: string[];
    num_samples: number;
    metric_configs?: Record<string, { data_fraction?: number; num_samples?: number }>;
    seed?: number;
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

export interface CheckpointEvaluationReport {
    checkpoint: string;
    date?: string | null;
    metrics: string[];
    metric_configs: Record<string, { data_fraction?: number; num_samples?: number }>;
    results?: CheckpointEvaluationResults;
}
