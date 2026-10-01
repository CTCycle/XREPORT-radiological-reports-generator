import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import {
  STARTUP_POLL_INTERVAL_MS,
  STARTUP_SLOW_THRESHOLD_MS,
  STARTUP_UNAVAILABLE_POLL_INTERVAL_MS,
  STARTUP_UNAVAILABLE_THRESHOLD_MS,
  StartupReadinessService,
} from './startup-readiness.service';

describe('StartupReadinessService', () => {
  let service: StartupReadinessService;
  let http: HttpTestingController;

  beforeEach(() => {
    vi.useFakeTimers();
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting(), StartupReadinessService],
    });
    service = TestBed.inject(StartupReadinessService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
    vi.useRealTimers();
  });

  async function settle(): Promise<void> {
    await Promise.resolve();
    await Promise.resolve();
  }

  async function respondToNextProbe(ok: boolean): Promise<void> {
    const request = http.expectOne('/api/health');
    if (ok) {
      request.flush({ status: 'ok' });
    } else {
      request.flush({ status: 'starting' }, { status: 503, statusText: 'Starting' });
    }
    await settle();
  }

  async function reachUnavailable(): Promise<void> {
    service.start();
    await respondToNextProbe(false);
    const seconds = STARTUP_UNAVAILABLE_THRESHOLD_MS / STARTUP_POLL_INTERVAL_MS;
    for (let second = 1; second <= seconds; second += 1) {
      vi.advanceTimersByTime(STARTUP_POLL_INTERVAL_MS);
      await respondToNextProbe(false);
    }
  }

  it('unlocks after the first valid health response', async () => {
    service.start();

    expect(service.phase()).toBe('starting');
    expect(service.attemptCount()).toBe(1);
    await respondToNextProbe(true);

    expect(service.phase()).toBe('ready');
    expect(service.lastSuccessfulProbe()).not.toBeNull();
    service.start();
    service.retry();
    http.expectNone('/api/health');
  });

  it('keeps polling through connection failures and recovers automatically', async () => {
    service.start();
    await respondToNextProbe(false);
    expect(service.phase()).toBe('starting');

    vi.advanceTimersByTime(STARTUP_POLL_INTERVAL_MS);
    await respondToNextProbe(true);

    expect(service.phase()).toBe('ready');
  });

  it('escalates to slow after the configured threshold', async () => {
    service.start();
    await respondToNextProbe(false);

    const seconds = STARTUP_SLOW_THRESHOLD_MS / STARTUP_POLL_INTERVAL_MS;
    for (let second = 1; second <= seconds; second += 1) {
      vi.advanceTimersByTime(STARTUP_POLL_INTERVAL_MS);
      await respondToNextProbe(false);
    }

    expect(service.phase()).toBe('slow');
  });

  it('escalates to unavailable and uses the slower background cadence', async () => {
    await reachUnavailable();

    expect(service.phase()).toBe('unavailable');
    vi.advanceTimersByTime(STARTUP_UNAVAILABLE_POLL_INTERVAL_MS);
    await respondToNextProbe(true);
    expect(service.phase()).toBe('ready');
  });

  it('resets the escalation window and probes immediately on retry', async () => {
    service.start();
    await respondToNextProbe(false);
    const seconds = STARTUP_SLOW_THRESHOLD_MS / STARTUP_POLL_INTERVAL_MS;
    for (let second = 1; second <= seconds; second += 1) {
      vi.advanceTimersByTime(STARTUP_POLL_INTERVAL_MS);
      await respondToNextProbe(false);
    }
    expect(service.phase()).toBe('slow');

    service.retry();
    expect(service.phase()).toBe('starting');
    await respondToNextProbe(false);

    for (let second = 1; second < seconds; second += 1) {
      vi.advanceTimersByTime(STARTUP_POLL_INTERVAL_MS);
      await respondToNextProbe(false);
    }
    expect(service.phase()).toBe('starting');
  });

  it('does not unlock on a malformed health response', async () => {
    service.start();
    const request = http.expectOne('/api/health');
    request.flush({ status: 'warming' });
    await settle();

    expect(service.phase()).toBe('starting');
    vi.advanceTimersByTime(STARTUP_POLL_INTERVAL_MS);
    await respondToNextProbe(true);
    expect(service.phase()).toBe('ready');
  });

  it('recovers automatically after entering unavailable', async () => {
    await reachUnavailable();

    expect(service.phase()).toBe('unavailable');
    vi.advanceTimersByTime(STARTUP_UNAVAILABLE_POLL_INTERVAL_MS);
    await respondToNextProbe(true);
    expect(service.phase()).toBe('ready');
    vi.advanceTimersByTime(STARTUP_UNAVAILABLE_POLL_INTERVAL_MS * 2);
    http.expectNone('/api/health');
  });
});
