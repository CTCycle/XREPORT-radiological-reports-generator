import { Injectable, inject } from '@angular/core';
import type {
  ArchitecturePreset,
  ArchitecturePresetRequest,
  ArchitecturePresetsResponse,
  ArchitecturePreviewRequest,
  ArchitecturePreviewResponse,
  CheckpointArchitectureResponse,
  EncodersResponse,
} from '../types/modelArchitectureApi';
import { ApiRequestService } from './api-request.service';

@Injectable({ providedIn: 'root' })
export class ModelArchitectureApiService {
  private readonly request = inject(ApiRequestService);

  getEncoders() { return this.request.request<EncodersResponse>('GET', '/api/model/encoders'); }
  getPresets() { return this.request.request<ArchitecturePresetsResponse>('GET', '/api/model/presets'); }
  createPreset(preset: ArchitecturePresetRequest) { return this.request.request<ArchitecturePreset>('POST', '/api/model/presets', preset); }
  updatePreset(key: string, preset: ArchitecturePresetRequest) { return this.request.request<ArchitecturePreset>('PUT', `/api/model/presets/${encodeURIComponent(key)}`, preset); }
  restorePresets() { return this.request.request<ArchitecturePresetsResponse>('POST', '/api/model/presets/restore'); }
  deletePreset(key: string) { return this.request.request<void>('DELETE', `/api/model/presets/${encodeURIComponent(key)}`); }
  preview(request: ArchitecturePreviewRequest) { return this.request.request<ArchitecturePreviewResponse>('POST', '/api/model/preview', request); }
  getCheckpointArchitecture(checkpoint: string) { return this.request.request<CheckpointArchitectureResponse>('GET', `/api/model/checkpoints/${encodeURIComponent(checkpoint)}/architecture`); }
}
