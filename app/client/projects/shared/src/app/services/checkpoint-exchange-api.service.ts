import { Injectable, inject } from '@angular/core';
import { ApiRequestService } from './api-request.service';
import type { JobStartResponse } from '../types/jobs';

@Injectable({ providedIn: 'root' })
export class CheckpointExchangeApiService {
  private readonly request = inject(ApiRequestService);

  startImport(file: File, renameTo?: string) {
    const form = new FormData();
    form.append('file', file, file.name);
    if (renameTo) form.append('rename_to', renameTo);
    return this.request.request<JobStartResponse>('POST', '/api/exchange/import', form);
  }

  startExport(checkpoint: string, includeEncoder: boolean) {
    return this.request.request<JobStartResponse>('POST', '/api/exchange/export', { checkpoint, include_encoder: includeEncoder });
  }

  exportUrl(filename: string): string {
    return `/api/exchange/exports/${encodeURIComponent(filename)}`;
  }
}
