import { DOCUMENT, isPlatformBrowser } from '@angular/common';
import { DestroyRef, Injectable, PLATFORM_ID, computed, inject, signal } from '@angular/core';
import { NotificationService } from './notification.service';
import { SettingsApiService } from './settings-api.service';
import {
  DEFAULT_THEME_PREFERENCE,
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
  private readonly settingsApi = inject(SettingsApiService);
  private readonly notifications = inject(NotificationService);
  private readonly systemPrefersDark = signal(false);
  private mediaQuery: MediaQueryList | null = null;
  private mediaQueryListener: ((event: MediaQueryListEvent) => void) | null = null;
  private mediaQuerySubscribed = false;
  private preferenceRevision = 0;
  private persistenceQueue = Promise.resolve();
  private persistedPreference: ThemePreference = DEFAULT_THEME_PREFERENCE;

  readonly preference = signal<ThemePreference>(DEFAULT_THEME_PREFERENCE);
  readonly persistenceError = signal<string | null>(null);
  readonly databaseSwitching = signal(false);
  readonly resolvedTheme = computed<ResolvedTheme>(() => {
    const preference = this.preference();
    return preference === 'system' ? (this.systemPrefersDark() ? 'dark' : 'light') : preference;
  });

  constructor() {
    this.persistedPreference = DEFAULT_THEME_PREFERENCE;
    this.preference.set(this.persistedPreference);
    this.initializeSystemPreference();
    this.applyResolvedTheme(this.resolvedTheme());
  }

  setPreference(preference: ThemePreference): Promise<void> {
    if (!isThemePreference(preference) || this.databaseSwitching()) return Promise.resolve();
    const revision = ++this.preferenceRevision;
    this.applyPreference(preference);
    this.persistenceError.set(null);
    const pending = this.persistenceQueue.then(async () => {
      if (revision !== this.preferenceRevision) return;
      const response = await this.settingsApi.updateSettings({ theme_preference: preference });
      if (revision !== this.preferenceRevision) return;
      if (!response.result) {
        this.reportSaveFailure(response.error ?? 'Unable to save the theme preference.');
        this.applyPreference(this.persistedPreference);
        return;
      }
      this.persistedPreference = response.result.values.theme_preference ?? DEFAULT_THEME_PREFERENCE;
      this.persistenceError.set(null);
      this.applyPreference(this.persistedPreference);
    });
    this.persistenceQueue = pending.catch(() => undefined);
    return pending.catch((error: unknown) => {
      if (revision !== this.preferenceRevision) return;
      this.reportSaveFailure(error instanceof Error ? error.message : 'Unable to save the theme preference.');
      this.applyPreference(this.persistedPreference);
    });
  }

  async loadFromDatabase(): Promise<void> {
    const revision = ++this.preferenceRevision;
    const response = await this.settingsApi.getSettings();
    if (revision !== this.preferenceRevision) return;
    if (!response.result) {
      this.persistenceError.set(response.error ?? 'Unable to load the saved theme preference.');
      return;
    }

    this.persistenceError.set(null);
    this.applyDatabasePreference(response.result.values.theme_preference);
  }

  applyDatabasePreference(preference: ThemePreference | null): void {
    if (preference !== null && !isThemePreference(preference)) return;
    this.preferenceRevision += 1;
    this.persistedPreference = preference ?? DEFAULT_THEME_PREFERENCE;
    this.persistenceError.set(null);
    this.applyPreference(this.persistedPreference);
  }

  setDatabaseSwitching(switching: boolean): void {
    this.databaseSwitching.set(switching);
  }

  async flushPersistence(): Promise<void> {
    await this.persistenceQueue;
  }

  private reportSaveFailure(message: string): void {
    this.persistenceError.set(message);
    this.notifications.notify({ type: 'warning', message });
  }

  private applyPreference(preference: ThemePreference): void {
    this.preference.set(preference);
    this.updateMediaQuerySubscription();
    this.applyResolvedTheme(this.resolvedTheme());
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
      this.mediaQuery.addEventListener('change', this.mediaQueryListener);
      this.mediaQuerySubscribed = true;
      return;
    }

    if (this.preference() !== 'system') this.removeMediaQueryListener();
  }

  private removeMediaQueryListener(): void {
    if (!this.mediaQuery || !this.mediaQueryListener || !this.mediaQuerySubscribed) return;
    this.mediaQuery.removeEventListener('change', this.mediaQueryListener);
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
