export type GenerationProfile = 'deterministic' | 'concise' | 'detailed';
export type ModelStatus = 'ready' | 'not_installed' | 'unvalidated' | 'downloading' | 'gated' | 'runtime_unavailable' | 'incompatible' | 'disabled';
export type LocalModelState = 'not_downloaded' | 'downloading' | 'downloaded_unvalidated' | 'ready' | 'failed';
export type OutputSection = 'raw_report' | 'findings' | 'impression';
export type ValidationStatus = 'blocked' | 'incompatible' | 'disabled' | 'pending' | 'degraded' | 'passed';

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
    origin: 'public' | 'custom';
    access_policy: 'open' | 'gated';
    access_url: string | null;
    anatomy_coverage: string;
    coverage_note: string | null;
    hardware_demand: 'low' | 'moderate' | 'high' | 'very_high';
    parameter_label: string | null;
    parameter_size: string | null;
    download_size_bytes: number | null;
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
    processor_files: string[];
    processor_target_prefix: string;
    required_files: string[];
    weight_file_sets: string[][];
    installation_state: 'not_installed' | 'staged' | 'active' | 'corrupt' | 'failed' | 'downloading';
    local_state: LocalModelState;
    can_download: boolean;
    can_delete_local: boolean;
    local_path: string | null;
    active_revision: string | null;
    candidate_revision: string | null;
    integrity_status: string;
    cloud_assessment: {
        checked_at?: string;
        source?: string;
        free_cloud_available?: boolean;
        reason?: string;
        error?: string | null;
    } | null;
    update_available: boolean;
    available_actions: string[];
}

export interface InferenceModelsResponse {
    models: ModelAvailability[];
    providers: Record<string, { status: ModelStatus; message: string | null }>;
}

export interface ModelUpdateCheckResponse {
    model_ref: string;
    repository_id: string;
    installed_revision: string | null;
    latest_revision: string | null;
    update_available: boolean;
    source: string;
    checked_at: string;
    error: string | null;
}

export type ModelMaintenanceAction = 'download' | 'repair' | 'reinstall' | 'download_update' | 'delete_local';

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
