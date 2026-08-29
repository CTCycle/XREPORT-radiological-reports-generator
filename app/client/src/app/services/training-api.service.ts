import { Injectable, inject } from '@angular/core';
import {
  CheckpointMetadataResponse,
  CheckpointsResponse,
  DeleteResponse,
  StartTrainingConfig,
} from '../types/trainingApi';
import { JobStartResponse } from '../types/jobs';
import { ApiRequestService } from './api-request.service';

@Injectable({ providedIn: 'root' })
export class TrainingApiService {
  private readonly request = inject(ApiRequestService);

  getCheckpoints() { return this.request.request<CheckpointsResponse>('GET', '/api/training/checkpoints'); }
  getCheckpointMetadata(checkpoint: string) { return this.request.request<CheckpointMetadataResponse>('GET', `/api/training/checkpoints/${encodeURIComponent(checkpoint)}/metadata`); }
  deleteCheckpoint(checkpoint: string) { return this.request.request<DeleteResponse>('DELETE', `/api/training/checkpoints/${encodeURIComponent(checkpoint)}`); }
  start(config: StartTrainingConfig) { return this.request.request<JobStartResponse>('POST', '/api/training/start', config); }
  resume(checkpoint: string, additionalEpochs: number) { return this.request.request<JobStartResponse>('POST', '/api/training/resume', { checkpoint, additional_epochs: additionalEpochs }); }
}
