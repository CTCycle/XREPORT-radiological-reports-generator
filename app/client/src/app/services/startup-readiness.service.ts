import { HttpClient } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';
import { firstValueFrom, timeout } from 'rxjs';
import { markStartupPhase } from './startup-timing';

export type StartupPhase = 'starting' | 'slow' | 'unavailable' | 'ready';

export interface HealthResponse {
  status?: string;
  [key: string]: unknown;
}

export const STARTUP_POLL_INTERVAL_MS = 1_000;
export const STARTUP_UNAVAILABLE_POLL_INTERVAL_MS = 3_000;
export const STARTUP_REQUEST_TIMEOUT_MS = 2_000;
export const STARTUP_SLOW_THRESHOLD_MS = 15_000;
export const STARTUP_UNAVAILABLE_THRESHOLD_MS = 60_000;

@Injectable({ providedIn: 'root' })
export class StartupReadinessService {
  private readonly http = inject(HttpClient);
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private requestInFlight = false;
  private started = false;
  private completed = false;

  readonly phase = signal<StartupPhase>('starting');
  readonly attemptCount = signal(0);
  readonly startedAt = signal<number | null>(null);
  readonly lastSuccessfulProbe = signal<number | null>(null);

  start(): void {
    if (this.started || this.completed) return;

    this.started = true;
    this.attemptCount.set(0);
    this.resetEscalationWindow();
    this.phase.set('starting');
    markStartupPhase('startup_screen_visible');
    void this.probe();
  }

  retry(): void {
    if (this.completed) return;
    if (!this.started) {
      this.start();
      return;
    }

    this.clearPollTimer();
    this.attemptCount.set(0);
    this.resetEscalationWindow();
    this.phase.set('starting');
    void this.probe();
  }

  private async probe(): Promise<void> {
    if (this.completed || this.requestInFlight) return;

    this.requestInFlight = true;
    this.attemptCount.update((attempts) => attempts + 1);
    try {
      const response = await firstValueFrom(
        this.http
          .get<HealthResponse>('/api/health')
          .pipe(timeout({ first: STARTUP_REQUEST_TIMEOUT_MS })),
      );

      if (response?.status === 'ok') {
        this.complete();
        return;
      }
    } catch {
      // Connection refusal and request timeouts are expected while the local service starts.
    } finally {
      this.requestInFlight = false;
    }

    if (this.completed) return;
    this.updatePhaseFromElapsedTime();
    this.scheduleNextProbe();
  }

  private complete(): void {
    if (this.completed) return;

    this.completed = true;
    this.clearPollTimer();
    this.lastSuccessfulProbe.set(Date.now());
    this.phase.set('ready');
    markStartupPhase('backend_ready');
  }

  private updatePhaseFromElapsedTime(): void {
    const startedAt = this.startedAt();
    if (startedAt === null) return;

    const elapsed = Math.max(0, Date.now() - startedAt);
    if (elapsed >= STARTUP_UNAVAILABLE_THRESHOLD_MS) {
      if (this.phase() !== 'unavailable') {
        this.phase.set('unavailable');
        markStartupPhase('startup_unavailable');
      }
      return;
    }

    if (elapsed >= STARTUP_SLOW_THRESHOLD_MS && this.phase() === 'starting') {
      this.phase.set('slow');
      markStartupPhase('startup_slow');
    }
  }

  private scheduleNextProbe(): void {
    this.clearPollTimer();
    const interval = this.phase() === 'unavailable'
      ? STARTUP_UNAVAILABLE_POLL_INTERVAL_MS
      : STARTUP_POLL_INTERVAL_MS;
    this.pollTimer = setTimeout(() => {
      this.pollTimer = null;
      void this.probe();
    }, interval);
  }

  private resetEscalationWindow(): void {
    this.startedAt.set(Date.now());
  }

  private clearPollTimer(): void {
    if (this.pollTimer === null) return;
    clearTimeout(this.pollTimer);
    this.pollTimer = null;
  }
}
