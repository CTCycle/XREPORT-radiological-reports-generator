import { Injectable, inject } from '@angular/core';
import { ApiRequestService } from './api-request.service';
import {
  GenerationProfile,
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
}
