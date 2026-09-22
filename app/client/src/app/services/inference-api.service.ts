import { Injectable, inject } from '@angular/core';
import { ApiRequestService } from './api-request.service';
import {
  GenerationProfile,
  InferenceHistoryDeleteResponse,
  InferenceHistoryDetail,
  InferenceHistoryResponse,
  InferenceHistorySort,
  InferenceHistoryStatus,
  InferenceHistoryUpdateRequest,
  InferenceModelsResponse,
  ModelMaintenanceAction,
  ModelUpdateCheckResponse,
} from '../types/inferenceApi';
import { JobStartResponse } from '../types/jobs';

@Injectable({ providedIn: 'root' })
export class InferenceApiService {
  private readonly request = inject(ApiRequestService);

  getModels() { return this.request.request<InferenceModelsResponse>('GET', '/api/inference/models'); }
  checkModelUpdate(modelRef: string) { return this.request.request<ModelUpdateCheckResponse>('POST', '/api/inference/models/check-update', { model_ref: modelRef }); }
  maintainModel(modelRef: string, action: ModelMaintenanceAction, revision?: string) {
    return this.request.request<JobStartResponse>('POST', '/api/inference/models/maintenance', { model_ref: modelRef, action, revision });
  }
  generateReports(images: File[], modelRef: string, profile: GenerationProfile, clinicalContext: string) {
    const form = new FormData();
    form.append('model_ref', modelRef);
    form.append('generation_profile', profile);
    form.append('clinical_context', clinicalContext);
    images.forEach((image) => form.append('images', image));
    return this.request.request<JobStartResponse>('POST', '/api/inference/generate', form);
  }
  listHistory(options: { modelRef?: string; status?: InferenceHistoryStatus; sort?: InferenceHistorySort; limit?: number; offset?: number } = {}) {
    const query = new URLSearchParams();
    if (options.modelRef) query.set('model_ref', options.modelRef);
    if (options.status) query.set('status', options.status);
    if (options.sort) query.set('sort', options.sort);
    if (options.limit !== undefined) query.set('limit', String(options.limit));
    if (options.offset !== undefined) query.set('offset', String(options.offset));
    const suffix = query.toString() ? `?${query.toString()}` : '';
    return this.request.request<InferenceHistoryResponse>('GET', `/api/inference/history${suffix}`);
  }
  getHistory(requestId: string) {
    return this.request.request<InferenceHistoryDetail>('GET', `/api/inference/history/${encodeURIComponent(requestId)}`);
  }
  updateHistory(requestId: string, request: InferenceHistoryUpdateRequest) {
    return this.request.request<InferenceHistoryDetail>('PATCH', `/api/inference/history/${encodeURIComponent(requestId)}`, request);
  }
  deleteHistory(requestId: string) {
    return this.request.request<InferenceHistoryDeleteResponse>('DELETE', `/api/inference/history/${encodeURIComponent(requestId)}`);
  }
}
