import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { NotificationService } from './notification.service';

describe('NotificationService', () => {
  let service: NotificationService;

  beforeEach(() => {
    localStorage.clear();
    vi.useFakeTimers();
    TestBed.configureTestingModule({});
    service = TestBed.inject(NotificationService);
  });

  afterEach(() => vi.useRealTimers());

  it('tracks unread notifications and supports read, dismiss and clear', () => {
    service.notify({ type: 'success', message: 'one', toast: false });
    service.notify({ type: 'error', message: 'two', toast: false });
    expect(service.unreadCount()).toBe(2);

    service.markRead(service.notifications()[0].id);
    expect(service.unreadCount()).toBe(1);

    service.dismiss(service.notifications()[0].id);
    expect(service.notifications().map((item) => item.message)).toEqual(['one']);

    service.clear();
    expect(service.notifications()).toEqual([]);
  });

  it('auto-dismisses toasts and can skip the history', () => {
    service.notify({ type: 'info', message: 'transient', history: false });
    expect(service.toasts()).toHaveLength(1);
    expect(service.notifications()).toEqual([]);

    vi.advanceTimersByTime(4000);
    expect(service.toasts()).toEqual([]);
  });

  it('restores history from storage', () => {
    service.notify({ type: 'warning', message: 'kept', toast: false });
    TestBed.resetTestingModule();
    expect(TestBed.inject(NotificationService).notifications()[0].message).toBe('kept');
  });
});
