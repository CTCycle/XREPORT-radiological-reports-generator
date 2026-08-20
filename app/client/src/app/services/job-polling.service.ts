import { Injectable } from '@angular/core';
import { Observable, catchError, defer, from, of, switchMap, takeWhile, timer } from 'rxjs';
import { ApiResult } from './api-request.service';
import { JobStatusResponse } from '../types/jobs';

@Injectable({ providedIn: 'root' })
export class JobPollingService {
  poll(fetchStatus: (jobId: string) => Promise<ApiResult<JobStatusResponse>>, jobId: string, intervalSeconds = 2): Observable<JobStatusResponse> {
    const intervalMs = Math.max(250, intervalSeconds * 1000);
    return timer(0, intervalMs).pipe(
      switchMap(() => defer(() => from(fetchStatus(jobId)))),
      switchMap((response) => response.error ? of({ job_id: jobId, job_type: 'unknown', status: 'failed' as const, progress: 0, result: null, error: response.error }) : response.result ? of(response.result) : of({ job_id: jobId, job_type: 'unknown', status: 'failed' as const, progress: 0, result: null, error: 'No result returned' })),
      takeWhile((status) => !['completed', 'failed', 'cancelled'].includes(status.status), true),
      catchError((error: unknown) => of({ job_id: jobId, job_type: 'unknown', status: 'failed' as const, progress: 0, result: null, error: error instanceof Error ? error.message : String(error) })),
    );
  }
}
