import { Injectable, inject } from '@angular/core';
import { ApiRequestService } from './api-request.service';
import { JobCancelResponse, JobListResponse, JobStatusResponse } from '../types/jobs';

@Injectable({ providedIn: 'root' })
export class JobsApiService {
  private readonly request = inject(ApiRequestService);

  list(jobType?: string, status?: string) {
    const params = new URLSearchParams();
    if (jobType) params.set('job_type', jobType);
    if (status) params.set('status', status);
    const query = params.toString();
    return this.request.request<JobListResponse>('GET', `/api/jobs${query ? `?${query}` : ''}`);
  }

  get(jobId: string) {
    return this.request.request<JobStatusResponse>('GET', `/api/jobs/${encodeURIComponent(jobId)}`);
  }

  cancel(jobId: string) {
    return this.request.request<JobCancelResponse>('DELETE', `/api/jobs/${encodeURIComponent(jobId)}`);
  }
}
