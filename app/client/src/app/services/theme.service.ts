import { DOCUMENT, isPlatformBrowser } from '@angular/common';
import { DestroyRef, Injectable, PLATFORM_ID, computed, inject, signal } from '@angular/core';
import { StorageService } from './storage.service';
import {
  DEFAULT_THEME_PREFERENCE,
  THEME_STORAGE_KEY,
  isThemePreference,
  type ResolvedTheme,
  type ThemePreference,
} from '../types/theme';

const DARK_MEDIA_QUERY = '(prefers-color-scheme: dark)';
const THEME_COLORS: Record<ResolvedTheme, string> = {
  light: '#e9eef6',
  dark: '#101419',
};

@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly document = inject(DOCUMENT);
  private readonly destroyRef = inject(DestroyRef);
  private readonly platformId = inject(PLATFORM_ID);
  private readonly storage = inject(StorageService);
  private readonly systemPrefersDark = signal(false);
  private mediaQuery: MediaQueryList | null = null;
  private mediaQueryListener: ((event: MediaQueryListEvent) => void) | null = null;
  private mediaQuerySubscribed = false;

  readonly preference = signal<ThemePreference>(DEFAULT_THEME_PREFERENCE);
  readonly resolvedTheme = computed<ResolvedTheme>(() => {
    const preference = this.preference();
    return preference === 'system' ? (this.systemPrefersDark() ? 'dark' : 'light') : preference;
  });

  constructor() {
    this.preference.set(this.readStoredPreference());
    this.initializeSystemPreference();
    this.applyResolvedTheme(this.resolvedTheme());
  }

  setPreference(preference: ThemePreference): void {
    if (!isThemePreference(preference)) return;
    this.preference.set(preference);
    this.storage.writeString(THEME_STORAGE_KEY, preference);
    this.updateMediaQuerySubscription();
    this.applyResolvedTheme(this.resolvedTheme());
  }

  private readStoredPreference(): ThemePreference {
    const stored = this.storage.readString(THEME_STORAGE_KEY);
    return isThemePreference(stored) ? stored : DEFAULT_THEME_PREFERENCE;
  }

  private initializeSystemPreference(): void {
    if (!isPlatformBrowser(this.platformId)) return;

    const view = this.document.defaultView;
    if (!view || typeof view.matchMedia !== 'function') return;

    let mediaQuery: MediaQueryList;
    try {
      mediaQuery = view.matchMedia(DARK_MEDIA_QUERY);
    } catch {
      return;
    }

    this.mediaQuery = mediaQuery;
    this.systemPrefersDark.set(mediaQuery.matches);
    this.mediaQueryListener = (event: MediaQueryListEvent) => {
      this.systemPrefersDark.set(event.matches);
      if (this.preference() === 'system') this.applyResolvedTheme(this.resolvedTheme());
    };

    this.updateMediaQuerySubscription();

    this.destroyRef.onDestroy(() => {
      this.removeMediaQueryListener();
      this.mediaQuery = null;
      this.mediaQueryListener = null;
    });
  }

  private updateMediaQuerySubscription(): void {
    if (!this.mediaQuery || !this.mediaQueryListener) return;

    if (this.preference() === 'system' && !this.mediaQuerySubscribed) {
      this.systemPrefersDark.set(this.mediaQuery.matches);
      let subscribed = false;
      if (typeof this.mediaQuery.addEventListener === 'function') {
        this.mediaQuery.addEventListener('change', this.mediaQueryListener);
        subscribed = true;
      } else if (typeof this.mediaQuery.addListener === 'function') {
        this.mediaQuery.addListener(this.mediaQueryListener);
        subscribed = true;
      }
      this.mediaQuerySubscribed = subscribed;
      return;
    }

    if (this.preference() !== 'system') this.removeMediaQueryListener();
  }

  private removeMediaQueryListener(): void {
    if (!this.mediaQuery || !this.mediaQueryListener || !this.mediaQuerySubscribed) return;
    if (typeof this.mediaQuery.removeEventListener === 'function') {
      this.mediaQuery.removeEventListener('change', this.mediaQueryListener);
    } else if (typeof this.mediaQuery.removeListener === 'function') {
      this.mediaQuery.removeListener(this.mediaQueryListener);
    }
    this.mediaQuerySubscribed = false;
  }

  private applyResolvedTheme(theme: ResolvedTheme): void {
    const root = this.document.documentElement;
    root.setAttribute('data-theme', theme);
    root.style.colorScheme = theme;

    const themeColorMeta = this.document.querySelector<HTMLMetaElement>('#theme-color-meta');
    themeColorMeta?.setAttribute('content', THEME_COLORS[theme]);
  }
}
