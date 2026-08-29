import { Injectable, inject } from '@angular/core';
import { readBoolean, readNumber, readString } from '../common/parsers';
import {
  BrowseResponse,
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
} from '../types/trainingApi';
import { JobStartResponse } from '../types/jobs';
import { ApiRequestService } from './api-request.service';

@Injectable({ providedIn: 'root' })
export class DatasetApiService {
  private readonly request = inject(ApiRequestService);

  getStatus() { return this.request.request<DatasetStatusResponse>('GET', '/api/preparation/dataset/status'); }
  getNames() { return this.request.request<DatasetNamesResponse>('GET', '/api/preparation/dataset/names'); }
  getProcessedNames() { return this.request.request<DatasetNamesResponse>('GET', '/api/preparation/dataset/processed/names'); }
  getProcessingMetadata(datasetName: string) { return this.request.request<ProcessingMetadataResponse>('GET', `/api/preparation/dataset/metadata/${encodeURIComponent(datasetName)}`); }
  deleteDataset(datasetName: string) { return this.request.request<DeleteResponse>('DELETE', `/api/preparation/dataset/${encodeURIComponent(datasetName)}`); }
  validateImagePath(folderPath: string) { return this.request.request<ImagePathResponse>('POST', '/api/preparation/images/validate', { folder_path: folderPath }); }
  uploadDataset(file: File) {
    const form = new FormData();
    form.append('file', file);
    return this.request.request<DatasetUploadResponse>('POST', '/api/upload/dataset', form);
  }
  loadDataset(request: LoadDatasetRequest) { return this.request.request<LoadDatasetResponse>('POST', '/api/preparation/dataset/load', request); }
  browseDirectory(path = '') { return this.request.request<BrowseResponse>('GET', path ? `/api/preparation/browse?path=${encodeURIComponent(path)}` : '/api/preparation/browse'); }
  processDataset(config: ProcessDatasetRequest) { return this.request.request<JobStartResponse>('POST', '/api/preparation/dataset/process', config); }
  getImageCount(datasetName: string) { return this.request.request<ImageCountResponse>('GET', `/api/preparation/dataset/${encodeURIComponent(datasetName)}/images/count`); }
  getImageMetadata(datasetName: string, index: number) { return this.request.request<ImageMetadataResponse>('GET', `/api/preparation/dataset/${encodeURIComponent(datasetName)}/images/${index}`); }
  getImageContentUrl(datasetName: string, index: number) { return `/api/preparation/dataset/${encodeURIComponent(datasetName)}/images/${index}/content`; }

  parseProcessDatasetResponse(result: Record<string, unknown>): ProcessDatasetResponse {
    return {
      success: readBoolean(result['success']) ?? true,
      total_samples: readNumber(result['total_samples']) ?? 0,
      train_samples: readNumber(result['train_samples']) ?? 0,
      validation_samples: readNumber(result['validation_samples']) ?? 0,
      vocabulary_size: readNumber(result['vocabulary_size']) ?? 0,
      message: readString(result['message']) ?? 'Dataset processed successfully',
    };
  }
}
