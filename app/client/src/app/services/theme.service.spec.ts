import { TestBed } from '@angular/core/testing';
import { StorageService } from './storage.service';
import { ThemeService } from './theme.service';
import { THEME_STORAGE_KEY } from '../types/theme';

type MediaListener = (event: MediaQueryListEvent) => void;

function createMediaQuery(initialMatches = false): {
  list: MediaQueryList;
  setMatches: (matches: boolean) => void;
  listenerCount: () => number;
} {
  let matches = initialMatches;
  const listeners = new Set<MediaListener>();
  const list = {
    get matches() { return matches; },
    media: '(prefers-color-scheme: dark)',
    onchange: null,
    addEventListener: (_type: string, listener: EventListenerOrEventListenerObject | null) => {
      if (typeof listener === 'function') listeners.add(listener as MediaListener);
    },
    removeEventListener: (_type: string, listener: EventListenerOrEventListenerObject | null) => {
      if (typeof listener === 'function') listeners.delete(listener as MediaListener);
    },
    addListener: (listener: MediaListener | null) => { if (listener) listeners.add(listener); },
    removeListener: (listener: MediaListener | null) => { if (listener) listeners.delete(listener); },
    dispatchEvent: () => true,
  } as unknown as MediaQueryList;

  return {
    list,
    setMatches: (nextMatches: boolean) => {
      matches = nextMatches;
      const event = { matches, media: '(prefers-color-scheme: dark)' } as MediaQueryListEvent;
      listeners.forEach((listener) => listener(event));
    },
    listenerCount: () => listeners.size,
  };
}

describe('ThemeService', () => {
  const originalMatchMedia = window.matchMedia;
  let mediaQuery: ReturnType<typeof createMediaQuery>;

  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.style.removeProperty('color-scheme');
    mediaQuery = createMediaQuery();
    Object.defineProperty(window, 'matchMedia', {
      configurable: true,
      value: () => mediaQuery.list,
    });
  });

  afterEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.style.removeProperty('color-scheme');
    Object.defineProperty(window, 'matchMedia', {
      configurable: true,
      value: originalMatchMedia,
    });
  });

  function createService(): ThemeService {
    TestBed.configureTestingModule({ providers: [StorageService, ThemeService] });
    return TestBed.inject(ThemeService);
  }

  it('defaults to system and resolves to the current light OS preference', () => {
    const service = createService();

    expect(service.preference()).toBe('system');
    expect(service.resolvedTheme()).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(mediaQuery.listenerCount()).toBe(1);
  });

  it('restores stored light, dark, and system preferences', () => {
    localStorage.setItem(THEME_STORAGE_KEY, 'light');
    expect(createService().preference()).toBe('light');

    TestBed.resetTestingModule();
    localStorage.setItem(THEME_STORAGE_KEY, 'dark');
    expect(createService().preference()).toBe('dark');

    TestBed.resetTestingModule();
    localStorage.setItem(THEME_STORAGE_KEY, 'system');
    expect(createService().preference()).toBe('system');
  });

  it('falls back to system for an invalid stored value', () => {
    localStorage.setItem(THEME_STORAGE_KEY, 'sepia');

    const service = createService();

    expect(service.preference()).toBe('system');
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('sepia');
  });

  it('switches themes immediately and persists the selected preference', () => {
    const service = createService();

    service.setPreference('dark');
    expect(service.resolvedTheme()).toBe('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(document.documentElement.style.colorScheme).toBe('dark');
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');

    service.setPreference('light');
    expect(service.resolvedTheme()).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('follows OS changes for system and ignores them for manual preferences', () => {
    const service = createService();

    mediaQuery.setMatches(true);
    expect(service.resolvedTheme()).toBe('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');

    service.setPreference('light');
    expect(mediaQuery.listenerCount()).toBe(0);
    mediaQuery.setMatches(false);
    expect(service.resolvedTheme()).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');

    service.setPreference('system');
    expect(mediaQuery.listenerCount()).toBe(1);
    expect(service.resolvedTheme()).toBe('light');
    mediaQuery.setMatches(true);
    expect(service.resolvedTheme()).toBe('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });
});
