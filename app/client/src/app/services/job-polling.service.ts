import { Injectable } from '@angular/core';
import { Observable, catchError, defer, from, of, switchMap, takeWhile, timer } from 'rxjs';
import { ApiResult } from './api-request.service';
import { JobStatusResponse } from '../types/jobs';

@Injectable({ providedIn: 'root' })
export class JobPollingService {
  poll(fetchStatus: (jobId: string) => Promise<ApiResult<JobStatusResponse>>, jobId: string, intervalSeconds = 2): Observable<JobStatusResponse> {
    const pollInterval = Math.max(0.25, intervalSeconds);
    const intervalMs = pollInterval * 1000;
    let lastStatus: JobStatusResponse | null = null;

    return timer(0, intervalMs).pipe(
      switchMap(() => defer(() => from(fetchStatus(jobId))).pipe(
        catchError((error: unknown) => of<ApiResult<JobStatusResponse>>({
          result: null,
          error: error instanceof Error ? error.message : String(error),
        })),
      )),
      switchMap((response) => {
        if (response.result && !response.error) {
          lastStatus = response.result;
          return of(response.result);
        }

        const error = response.error ?? 'No result returned';
        if (this.isMissingJobError(error)) return of(this.failureStatus(jobId, error, pollInterval));
        if (lastStatus) return of({ ...lastStatus, error });
        return of<JobStatusResponse>({
          job_id: jobId,
          job_type: 'unknown',
          status: 'pending',
          poll_interval: pollInterval,
          progress: 0,
          result: null,
          error,
        });
      }),
      takeWhile((status) => !['completed', 'failed', 'cancelled'].includes(status.status), true),
      catchError((error: unknown) => of(this.failureStatus(jobId, error instanceof Error ? error.message : String(error), pollInterval))),
    );
  }

  private isMissingJobError(error: string): boolean {
    return error.startsWith('404 ');
  }

  private failureStatus(jobId: string, error: string, pollInterval: number): JobStatusResponse {
    return {
      job_id: jobId,
      job_type: 'unknown',
      status: 'failed',
      poll_interval: pollInterval,
      progress: 0,
      result: null,
      error,
    };
  }
}
