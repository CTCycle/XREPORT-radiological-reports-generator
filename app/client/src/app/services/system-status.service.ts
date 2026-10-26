import { Injectable, inject, signal } from '@angular/core';
import type { SystemStatusResponse } from '../types/systemStatus';
import { ApiRequestService } from './api-request.service';
import { NotificationService } from './notification.service';

const POLL_INTERVAL_MS = 5000;

@Injectable({ providedIn: 'root' })
export class SystemStatusService {
  private readonly request = inject(ApiRequestService);
  private readonly notifications = inject(NotificationService);
  private timer: ReturnType<typeof setInterval> | null = null;
  private inFlight = false;
  private cudaReported = false;

  /** `null` until the first response arrives. */
  readonly backendOnline = signal<boolean | null>(null);
  readonly status = signal<SystemStatusResponse | null>(null);

  private readonly onVisibilityChange = (): void => {
    if (!document.hidden) void this.refresh();
  };

  start(): void {
    if (this.timer !== null) return;
    void this.refresh();
    this.timer = setInterval(() => { if (!document.hidden) void this.refresh(); }, POLL_INTERVAL_MS);
    document.addEventListener('visibilitychange', this.onVisibilityChange);
  }

  stop(): void {
    if (this.timer === null) return;
    clearInterval(this.timer);
    this.timer = null;
    document.removeEventListener('visibilitychange', this.onVisibilityChange);
  }

  async refresh(): Promise<void> {
    if (this.inFlight) return;
    this.inFlight = true;
    try {
      const { result } = await this.request.request<SystemStatusResponse>('GET', '/api/system/status');
      this.apply(result);
    } finally {
      this.inFlight = false;
    }
  }

  private apply(result: SystemStatusResponse | null): void {
    const previous = this.backendOnline();
    const online = result !== null;
    this.backendOnline.set(online);
    this.status.set(result);
    if (previous === true && !online) {
      this.notifications.notify({ type: 'warning', message: 'Backend connection lost.' });
    } else if (previous === false && online) {
      this.notifications.notify({ type: 'success', message: 'Backend connection restored.' });
    }
    if (result && !result.cuda.available && !this.cudaReported) {
      this.cudaReported = true;
      this.notifications.notify({ type: 'info', message: 'CUDA is unavailable; inference runs on the CPU.', toast: false });
    }
  }
}
