import type {
  AggregateStatistic as ApiAggregateStatistic,
  CheckpointEvaluationReportResponse as ApiCheckpointEvaluationReport,
  CheckpointEvaluationRequest as ApiCheckpointEvaluationRequest,
  CheckpointEvaluationResults as ApiCheckpointEvaluationResults,
  HistogramModel as ApiHistogramModel,
  RecordStatisticsResponse as ApiRecordStatisticsResponse,
  ValidationReportResponse as ApiValidationReportResponse,
  ValidationRequest as ApiValidationRequest,
  ValidationRunListResponse as ApiValidationRunListResponse,
  ValidationRunSummary as ApiValidationRunSummary,
} from './generated/studio/types.gen';

export type ValidationRequest = ApiValidationRequest;
export type ValidationReport = ApiValidationReportResponse;
export type ValidationRunList = ApiValidationRunListResponse;
export type ValidationRunSummary = ApiValidationRunSummary;
export type AggregateStatistic = ApiAggregateStatistic;
export type HistogramModel = ApiHistogramModel;
export type RecordStatisticsPage = ApiRecordStatisticsResponse;
export type RecordStatisticsRow = RecordStatisticsPage['rows'][number];
export type RecordKind = 'text' | 'image';

export interface RecordStatisticsQuery {
  kind: RecordKind;
  sort?: string;
  descending?: boolean;
  search?: string;
  limit?: number;
  offset?: number;
}

export type CheckpointEvaluationRequest = ApiCheckpointEvaluationRequest;
export type CheckpointEvaluationResults = ApiCheckpointEvaluationResults;
export type CheckpointEvaluationReport = ApiCheckpointEvaluationReport;
