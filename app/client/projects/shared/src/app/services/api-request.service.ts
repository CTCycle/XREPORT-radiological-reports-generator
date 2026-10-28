import { HttpClient, HttpErrorResponse, HttpEventType } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { firstValueFrom, lastValueFrom, tap } from 'rxjs';

export interface ApiResult<T> {
  result: T | null;
  error: string | null;
}

@Injectable({ providedIn: 'root' })
export class ApiRequestService {
  private readonly http = inject(HttpClient);

  async request<T>(method: string, url: string, body?: unknown, options: { keepalive?: boolean } = {}): Promise<ApiResult<T>> {
    try {
      const result = await firstValueFrom(this.http.request<T>(method, url, { body, ...options }));
      return { result, error: null };
    } catch (error) {
      return { result: null, error: this.formatError(error) };
    }
  }

  async uploadWithProgress<T>(url: string, body: FormData, onProgress: (percent: number) => void): Promise<ApiResult<T>> {
    try {
      const response = await lastValueFrom(this.http.post<T>(url, body, { reportProgress: true, observe: 'events' }).pipe(
        tap((event) => { if (event.type === HttpEventType.UploadProgress && event.total) onProgress(Math.round((event.loaded / event.total) * 100)); }),
      ));
      return { result: (response as { body?: T | null }).body ?? null, error: null };
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
}
