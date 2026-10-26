import type {
  BodyGenerateReportsApiInferenceGeneratePost as GenerateReportsBody,
  InferenceHistoryDeleteResponse as ApiInferenceHistoryDeleteResponse,
  InferenceHistoryDetail as ApiInferenceHistoryDetail,
  InferenceHistoryReport as ApiInferenceHistoryReport,
  InferenceHistoryReportSummary as ApiInferenceHistoryReportSummary,
  InferenceHistoryResponse as ApiInferenceHistoryResponse,
  InferenceHistorySummary as ApiInferenceHistorySummary,
  InferenceHistoryUpdateRequest as ApiInferenceHistoryUpdateRequest,
  InferenceModelsResponse as ApiInferenceModelsResponse,
  ModelAvailability as ApiModelAvailability,
  ModelMaintenanceRequest as ApiModelMaintenanceRequest,
  ModelUpdateCheckResponse as ApiModelUpdateCheckResponse,
  RuntimeReleaseResponse as ApiRuntimeReleaseResponse,
} from './generated/xreport/types.gen';

export type GenerationProfile = GenerateReportsBody['generation_profile'];
export type ModelStatus = ApiModelAvailability['status'];
export type OutputSection = NonNullable<ApiModelAvailability['output_sections']>[number];
export type ValidationStatus = ApiModelAvailability['validation_status'];
export type ModelAvailability = ApiModelAvailability;
export type InferenceModelsResponse = ApiInferenceModelsResponse;
export type ModelUpdateCheckResponse = ApiModelUpdateCheckResponse;
export type RuntimeReleaseResponse = ApiRuntimeReleaseResponse;
export type ModelMaintenanceAction = ApiModelMaintenanceRequest['action'];
export type InferenceHistoryStatus = ApiInferenceHistorySummary['status'];
export type InferenceHistorySort = 'newest' | 'oldest';
export type InferenceHistoryReportSummary = ApiInferenceHistoryReportSummary;
export type InferenceHistorySummary = ApiInferenceHistorySummary;
export type InferenceHistoryResponse = ApiInferenceHistoryResponse;
export type InferenceHistoryReport = ApiInferenceHistoryReport;
export type InferenceHistoryDetail = ApiInferenceHistoryDetail;
export type InferenceHistoryUpdateRequest = ApiInferenceHistoryUpdateRequest;
export type InferenceHistoryDeleteResponse = ApiInferenceHistoryDeleteResponse;
