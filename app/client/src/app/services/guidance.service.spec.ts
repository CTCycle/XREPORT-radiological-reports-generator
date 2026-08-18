import { TestBed } from '@angular/core/testing';
import { GuidanceService, GUIDANCE_STORAGE_KEY } from './guidance.service';
import { StorageService } from './storage.service';

describe('GuidanceService', () => {
  beforeEach(() => {
    localStorage.clear();
    TestBed.configureTestingModule({ providers: [StorageService, GuidanceService] });
  });

  it('shows new content, persists terminal status, and hides it afterward', () => {
    const service = TestBed.inject(GuidanceService);

    expect(service.shouldShow('inference-first-use', 1)).toBe(true);
    service.markSeen('inference-first-use', 1);
    expect(service.shouldShow('inference-first-use', 1)).toBe(false);
    expect(service.getEntry('inference-first-use')).toEqual({ version: 1, status: 'seen' });
    expect(JSON.parse(localStorage.getItem(GUIDANCE_STORAGE_KEY) ?? '{}')).toEqual({
      'inference-first-use': { version: 1, status: 'seen' },
    });
  });

  it.each([
    ['dismiss', 'dismissed'],
    ['skip', 'skipped'],
    ['complete', 'completed'],
  ] as const)('persists %s status', (operation, status) => {
    const service = TestBed.inject(GuidanceService);

    service[operation]('tour', 1);

    expect(service.getEntry('tour')).toEqual({ version: 1, status });
    expect(service.shouldShow('tour', 1)).toBe(false);
  });

  it('resurfaces only the content whose version changed', () => {
    const service = TestBed.inject(GuidanceService);

    service.dismiss('tip', 1);

    expect(service.shouldShow('tip', 1)).toBe(false);
    expect(service.shouldShow('tip', 2)).toBe(true);
    service.markSeen('tip', 2);
    expect(service.getEntry('tip')).toEqual({ version: 2, status: 'seen' });
  });

  it('recovers from malformed storage and supports manual replay requests', () => {
    localStorage.setItem(GUIDANCE_STORAGE_KEY, '{not-json');
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({ providers: [StorageService, GuidanceService] });
    const service = TestBed.inject(GuidanceService);
    let requested = '';
    service.tourRequests$.subscribe((tourId) => requested = tourId);

    expect(service.getEntry('tour')).toBeNull();
    service.complete('tour', 1);
    service.requestTour('tour');

    expect(requested).toBe('tour');
  });
});
