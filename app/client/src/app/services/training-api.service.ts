import { Injectable, inject } from '@angular/core';
import {
  CheckpointMetadataResponse,
  CheckpointsResponse,
  DeleteResponse,
  StartTrainingConfig,
  TrainingStatusResponse,
} from '../types/trainingApi';
import { JobCancelResponse, JobStartResponse, JobStatusResponse } from '../types/jobs';
import { ApiRequestService } from './api-request.service';

@Injectable({ providedIn: 'root' })
export class TrainingApiService {
  private readonly request = inject(ApiRequestService);

  getCheckpoints() { return this.request.request<CheckpointsResponse>('GET', '/api/training/checkpoints'); }
  getCheckpointMetadata(checkpoint: string) { return this.request.request<CheckpointMetadataResponse>('GET', `/api/training/checkpoints/${encodeURIComponent(checkpoint)}/metadata`); }
  deleteCheckpoint(checkpoint: string) { return this.request.request<DeleteResponse>('DELETE', `/api/training/checkpoints/${encodeURIComponent(checkpoint)}`); }
  getStatus() { return this.request.request<TrainingStatusResponse>('GET', '/api/training/status'); }
  start(config: StartTrainingConfig) { return this.request.request<JobStartResponse>('POST', '/api/training/start', config); }
  resume(checkpoint: string, additionalEpochs: number) { return this.request.request<JobStartResponse>('POST', '/api/training/resume', { checkpoint, additional_epochs: additionalEpochs }); }
  getJobStatus(jobId: string) { return this.request.request<JobStatusResponse>('GET', `/api/training/jobs/${encodeURIComponent(jobId)}`); }
  cancelJob(jobId: string) { return this.request.request<JobCancelResponse>('DELETE', `/api/training/jobs/${encodeURIComponent(jobId)}`); }
}
