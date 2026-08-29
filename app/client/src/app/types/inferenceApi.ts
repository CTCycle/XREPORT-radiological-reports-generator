import type { components } from './api.generated';

type Schemas = components['schemas'];

export type GenerationProfile = Schemas['Body_generate_reports_api_inference_generate_post']['generation_profile'];
export type ModelStatus = Schemas['ModelAvailability']['status'];
export type OutputSection = NonNullable<Schemas['ModelAvailability']['output_sections']>[number];
export type ValidationStatus = Schemas['ModelAvailability']['validation_status'];
export type ModelAvailability = Schemas['ModelAvailability'];
export type InferenceModelsResponse = Schemas['InferenceModelsResponse'];
export type ModelUpdateCheckResponse = Schemas['ModelUpdateCheckResponse'];
export type ModelMaintenanceAction = Schemas['ModelMaintenanceRequest']['action'];
export type CheckpointEvaluationRequest = Schemas['CheckpointEvaluationRequest'];
export type CheckpointEvaluationResults = Schemas['CheckpointEvaluationResults'];
export type CheckpointEvaluationReport = Schemas['CheckpointEvaluationReportResponse'];
