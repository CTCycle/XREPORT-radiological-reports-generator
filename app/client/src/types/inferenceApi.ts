export type GenerationProfile = 'deterministic' | 'concise' | 'detailed';
export type ModelStatus = 'ready' | 'not_installed' | 'unvalidated' | 'gated' | 'runtime_unavailable' | 'incompatible' | 'disabled';
export type OutputSection = 'raw_report' | 'findings' | 'impression';
export type ValidationStatus = 'blocked' | 'incompatible' | 'disabled' | 'pending' | 'passed';

export interface ModelAvailability {
    model_ref: string;
    provider: 'huggingface' | 'xreport';
    display_name: string;
    description: string;
    status: ModelStatus;
    status_message: string | null;
    enabled: boolean;
    validation_status: ValidationStatus;
    validation_message: string | null;
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
    model_loader: string | null;
    processor_loader: string | null;
    adapter: string | null;
    trust_remote_code: boolean;
    remote_code_approved: boolean;
    output_sections: OutputSection[];
    max_current_images: number;
    supports_prior_images: boolean;
    supports_clinical_context: boolean;
    preferred_dtype: string;
    quantization: string[];
    prompt_profile: string | null;
    license: string | null;
    resource_policy: {
        max_snapshot_size_bytes: number | null;
        reason: string | null;
    };
    runtime_constraints: {
        min_transformers: string | null;
        max_transformers_exclusive: string | null;
        required_modules: string[];
    };
    processor_repository_id: string | null;
    processor_revision: string | null;
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
