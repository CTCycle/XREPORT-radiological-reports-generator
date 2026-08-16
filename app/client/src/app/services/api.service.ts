import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { asRecord, readBoolean, readNumber, readNumberArray, readString } from '../common/parsers';
import {
  CheckpointEvaluationReport,
  CheckpointEvaluationRequest,
  GenerationProfile,
  InferenceModelsResponse,
  ModelMaintenanceAction,
  ModelUpdateCheckResponse,
} from '../types/inferenceApi';
import {
  BrowseResponse,
  CheckpointMetadataResponse,
  CheckpointsResponse,
  DatasetNamesResponse,
  DatasetStatusResponse,
  DatasetUploadResponse,
  DeleteResponse,
  ImageCountResponse,
  ImageMetadataResponse,
  ImagePathResponse,
  LoadDatasetRequest,
  LoadDatasetResponse,
  ProcessDatasetRequest,
  ProcessDatasetResponse,
  ProcessingMetadataResponse,
  StartTrainingConfig,
  TrainingStatusResponse,
} from '../types/trainingApi';
import {
  ImageStatistics,
  PixelDistribution,
  TextStatistics,
  ValidationReport,
  ValidationRequest,
  ValidationResponse,
} from '../types/validationApi';
import { JobCancelResponse, JobStartResponse, JobStatusResponse } from '../types/jobs';

export interface ApiResult<T> {
  result: T | null;
  error: string | null;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);

  private async request<T>(method: string, url: string, body?: unknown): Promise<ApiResult<T>> {
    try {
      const result = await firstValueFrom(this.http.request<T>(method, url, { body }));
      return { result, error: null };
    } catch (error) {
      return { result: null, error: this.formatError(error) };
    }
  }

  private formatError(error: unknown): string {
    if (error instanceof HttpErrorResponse) {
      const detail = typeof error.error === 'string' ? error.error : error.error?.detail ?? error.error;
      return `${error.status} ${error.statusText || 'Request failed'}${detail ? `: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}` : ''}`;
    }
    return error instanceof Error ? error.message : String(error);
  }

  getInferenceModels() { return this.request<InferenceModelsResponse>('GET', '/api/inference/models'); }
  checkInferenceModelUpdate(modelRef: string) { return this.request<ModelUpdateCheckResponse>('POST', '/api/inference/models/check-update', { model_ref: modelRef }); }
  maintainInferenceModel(modelRef: string, action: ModelMaintenanceAction, revision?: string) {
    return this.request<JobStartResponse>('POST', '/api/inference/models/maintenance', { model_ref: modelRef, action, revision });
  }
  generateReports(images: File[], modelRef: string, profile: GenerationProfile, clinicalContext: string) {
    const form = new FormData();
    form.append('model_ref', modelRef);
    form.append('generation_profile', profile);
    form.append('clinical_context', clinicalContext);
    images.forEach((image) => form.append('images', image));
    return this.request<JobStartResponse>('POST', '/api/inference/generate', form);
  }
  getInferenceJobStatus(jobId: string) { return this.request<JobStatusResponse>('GET', `/api/inference/jobs/${encodeURIComponent(jobId)}`); }
  cancelInferenceJob(jobId: string) { return this.request<JobCancelResponse>('DELETE', `/api/inference/jobs/${encodeURIComponent(jobId)}`); }
  evaluateCheckpoint(checkpoint: string, metrics: string[], numSamples = 10, metricConfigs?: Record<string, { data_fraction?: number; num_samples?: number }>, seed?: number) {
    const request: CheckpointEvaluationRequest = { checkpoint, metrics, num_samples: numSamples, metric_configs: metricConfigs, seed };
    return this.request<JobStartResponse>('POST', '/api/validation/checkpoint', request);
  }
  getCheckpointEvaluationJobStatus(jobId: string) { return this.request<JobStatusResponse>('GET', `/api/validation/jobs/${encodeURIComponent(jobId)}`); }
  getCheckpointEvaluationReport(checkpoint: string) { return this.request<CheckpointEvaluationReport>('GET', `/api/validation/checkpoint/reports/${encodeURIComponent(checkpoint)}`); }

  getDatasetStatus() { return this.request<DatasetStatusResponse>('GET', '/api/preparation/dataset/status'); }
  getDatasetNames() { return this.request<DatasetNamesResponse>('GET', '/api/preparation/dataset/names'); }
  getProcessedDatasetNames() { return this.request<DatasetNamesResponse>('GET', '/api/preparation/dataset/processed/names'); }
  getProcessingMetadata(datasetName: string) { return this.request<ProcessingMetadataResponse>('GET', `/api/preparation/dataset/metadata/${encodeURIComponent(datasetName)}`); }
  deleteDataset(datasetName: string) { return this.request<DeleteResponse>('DELETE', `/api/preparation/dataset/${encodeURIComponent(datasetName)}`); }
  validateImagePath(folderPath: string) { return this.request<ImagePathResponse>('POST', '/api/preparation/images/validate', { folder_path: folderPath }); }
  uploadDataset(file: File) { const form = new FormData(); form.append('file', file); return this.request<DatasetUploadResponse>('POST', '/api/upload/dataset', form); }
  loadDataset(request: LoadDatasetRequest) { return this.request<LoadDatasetResponse>('POST', '/api/preparation/dataset/load', request); }
  browseDirectory(path = '') { return this.request<BrowseResponse>('GET', path ? `/api/preparation/browse?path=${encodeURIComponent(path)}` : '/api/preparation/browse'); }
  processDataset(config: ProcessDatasetRequest) { return this.request<JobStartResponse>('POST', '/api/preparation/dataset/process', config); }
  getPreparationJobStatus(jobId: string) { return this.request<JobStatusResponse>('GET', `/api/preparation/jobs/${encodeURIComponent(jobId)}`); }
  cancelPreparationJob(jobId: string) { return this.request<JobCancelResponse>('DELETE', `/api/preparation/jobs/${encodeURIComponent(jobId)}`); }
  getDatasetImageCount(datasetName: string) { return this.request<ImageCountResponse>('GET', `/api/preparation/dataset/${encodeURIComponent(datasetName)}/images/count`); }
  getDatasetImageMetadata(datasetName: string, index: number) { return this.request<ImageMetadataResponse>('GET', `/api/preparation/dataset/${encodeURIComponent(datasetName)}/images/${index}`); }
  getDatasetImageContentUrl(datasetName: string, index: number) { return `/api/preparation/dataset/${encodeURIComponent(datasetName)}/images/${index}/content`; }

  getCheckpoints() { return this.request<CheckpointsResponse>('GET', '/api/training/checkpoints'); }
  getCheckpointMetadata(checkpoint: string) { return this.request<CheckpointMetadataResponse>('GET', `/api/training/checkpoints/${encodeURIComponent(checkpoint)}/metadata`); }
  deleteCheckpoint(checkpoint: string) { return this.request<DeleteResponse>('DELETE', `/api/training/checkpoints/${encodeURIComponent(checkpoint)}`); }
  getTrainingStatus() { return this.request<TrainingStatusResponse>('GET', '/api/training/status'); }
  startTraining(config: StartTrainingConfig) { return this.request<JobStartResponse>('POST', '/api/training/start', config); }
  resumeTraining(checkpoint: string, additionalEpochs: number) { return this.request<JobStartResponse>('POST', '/api/training/resume', { checkpoint, additional_epochs: additionalEpochs }); }
  getTrainingJobStatus(jobId: string) { return this.request<JobStatusResponse>('GET', `/api/training/jobs/${encodeURIComponent(jobId)}`); }
  cancelTrainingJob(jobId: string) { return this.request<JobCancelResponse>('DELETE', `/api/training/jobs/${encodeURIComponent(jobId)}`); }

  runValidation(request: ValidationRequest) { return this.request<JobStartResponse>('POST', '/api/validation/run', request); }
  getValidationReport(datasetName: string) { return this.request<ValidationReport>('GET', `/api/validation/reports/${encodeURIComponent(datasetName)}`); }
  getValidationJobStatus(jobId: string) { return this.request<JobStatusResponse>('GET', `/api/validation/jobs/${encodeURIComponent(jobId)}`); }
  cancelValidationJob(jobId: string) { return this.request<JobCancelResponse>('DELETE', `/api/validation/jobs/${encodeURIComponent(jobId)}`); }

  parseValidationResponse(result: Record<string, unknown>): ValidationResponse {
    const pixel = asRecord(result['pixel_distribution']);
    const image = asRecord(result['image_statistics']);
    const text = asRecord(result['text_statistics']);
    const pixelDistribution: PixelDistribution | undefined = pixel && readNumberArray(pixel['bins']) && readNumberArray(pixel['counts'])
      ? { bins: readNumberArray(pixel['bins'])!, counts: readNumberArray(pixel['counts'])! } : undefined;
    const imageStatistics: ImageStatistics | undefined = image ? this.parseImageStatistics(image) : undefined;
    const textStatistics: TextStatistics | undefined = text ? this.parseTextStatistics(text) : undefined;
    return { success: readBoolean(result['success']) ?? true, message: readString(result['message']) ?? 'Validation completed', pixel_distribution: pixelDistribution, image_statistics: imageStatistics, text_statistics: textStatistics };
  }

  parseProcessDatasetResponse(result: Record<string, unknown>): ProcessDatasetResponse {
    return { success: readBoolean(result['success']) ?? true, total_samples: readNumber(result['total_samples']) ?? 0, train_samples: readNumber(result['train_samples']) ?? 0, validation_samples: readNumber(result['validation_samples']) ?? 0, vocabulary_size: readNumber(result['vocabulary_size']) ?? 0, message: readString(result['message']) ?? 'Dataset processed successfully' };
  }

  private parseImageStatistics(payload: Record<string, unknown>): ImageStatistics | undefined {
    const values = ['count', 'mean_height', 'mean_width', 'mean_pixel_value', 'std_pixel_value', 'mean_noise_std', 'mean_noise_ratio'].map((key) => readNumber(payload[key]));
    return values.every((value): value is number => value !== undefined) ? { count: values[0], mean_height: values[1], mean_width: values[2], mean_pixel_value: values[3], std_pixel_value: values[4], mean_noise_std: values[5], mean_noise_ratio: values[6] } : undefined;
  }

  private parseTextStatistics(payload: Record<string, unknown>): TextStatistics | undefined {
    const values = ['count', 'total_words', 'unique_words', 'avg_words_per_report', 'min_words_per_report', 'max_words_per_report'].map((key) => readNumber(payload[key]));
    return values.every((value): value is number => value !== undefined) ? { count: values[0], total_words: values[1], unique_words: values[2], avg_words_per_report: values[3], min_words_per_report: values[4], max_words_per_report: values[5] } : undefined;
  }
}
