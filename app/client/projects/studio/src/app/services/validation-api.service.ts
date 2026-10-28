import { Injectable, inject } from '@angular/core';
import type {
  CheckpointEvaluationReport,
  CheckpointEvaluationRequest,
  RecordStatisticsPage,
  RecordStatisticsQuery,
  ValidationReport,
  ValidationRequest,
  ValidationRunList,
} from '../types/validationApi';
import { JobStartResponse } from '@xreport/shared/app/types/jobs';
import { ApiRequestService } from '@xreport/shared/app/services/api-request.service';

@Injectable({ providedIn: 'root' })
export class ValidationApiService {
  private readonly request = inject(ApiRequestService);

  evaluateCheckpoint(checkpoint: string, metrics: string[], numSamples = 10, metricConfigs?: Record<string, { data_fraction?: number; num_samples?: number }>, seed?: number) {
    const evaluationRequest: CheckpointEvaluationRequest = { checkpoint, metrics, num_samples: numSamples, metric_configs: metricConfigs, seed };
    return this.request.request<JobStartResponse>('POST', '/api/validation/checkpoint', evaluationRequest);
  }
  getCheckpointEvaluationReport(checkpoint: string) { return this.request.request<CheckpointEvaluationReport>('GET', `/api/validation/checkpoint/reports/${encodeURIComponent(checkpoint)}`); }
  run(request: ValidationRequest) { return this.request.request<JobStartResponse>('POST', '/api/validation/run', request); }
  listRuns(datasetName: string) { return this.request.request<ValidationRunList>('GET', `/api/validation/runs/${encodeURIComponent(datasetName)}`); }
  getRun(datasetName: string, runId: number) { return this.request.request<ValidationReport>('GET', `/api/validation/runs/${encodeURIComponent(datasetName)}/${runId}`); }
  getRecords(datasetName: string, runId: number, query: RecordStatisticsQuery) {
    const params = new URLSearchParams({ kind: query.kind, limit: String(query.limit ?? 50), offset: String(query.offset ?? 0), descending: String(query.descending ?? false) });
    if (query.sort) params.set('sort', query.sort);
    if (query.search) params.set('search', query.search);
    return this.request.request<RecordStatisticsPage>('GET', `/api/validation/runs/${encodeURIComponent(datasetName)}/${runId}/records?${params.toString()}`);
  }
}
