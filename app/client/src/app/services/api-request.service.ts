import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';

export interface ApiResult<T> {
  result: T | null;
  error: string | null;
}

@Injectable({ providedIn: 'root' })
export class ApiRequestService {
  private readonly http = inject(HttpClient);

  async request<T>(method: string, url: string, body?: unknown): Promise<ApiResult<T>> {
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
}
