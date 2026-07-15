export type GenerationProfile = 'deterministic' | 'concise' | 'detailed';
export type ModelStatus = 'ready' | 'not_installed' | 'gated' | 'runtime_unavailable' | 'incompatible' | 'disabled';

export interface ModelAvailability {
    model_ref: string;
    provider: 'ollama' | 'huggingface' | 'xreport' | 'maira2';
    display_name: string;
    description: string;
    status: ModelStatus;
    category: string;
    recommended: boolean;
    research_only: boolean;
    gated: boolean;
    parameter_size: string | null;
    local_size_bytes: number | null;
    input_semantics: 'single_image' | 'independent_images' | 'single_study';
    capabilities: {
        clinical_context: boolean;
        prior_report: boolean;
        multiple_current_views: boolean;
        findings: boolean;
        impression: boolean;
        grounding: boolean;
    };
    model_revision: string | null;
}

export interface InferenceModelsResponse {
    models: ModelAvailability[];
    providers: Record<string, { status: ModelStatus; message: string | null }>;
}

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
