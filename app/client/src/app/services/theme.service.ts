import { DOCUMENT, isPlatformBrowser } from '@angular/common';
import { DestroyRef, Injectable, PLATFORM_ID, computed, inject, signal } from '@angular/core';
import { StorageService } from './storage.service';
import { SettingsApiService } from './settings-api.service';
import {
  DEFAULT_THEME_PREFERENCE,
  LEGACY_THEME_STORAGE_KEY,
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
  private readonly settingsApi = inject(SettingsApiService);
  private readonly systemPrefersDark = signal(false);
  private mediaQuery: MediaQueryList | null = null;
  private mediaQueryListener: ((event: MediaQueryListEvent) => void) | null = null;
  private mediaQuerySubscribed = false;
  private preferenceRevision = 0;
  private persistenceQueue = Promise.resolve();
  private persistedPreference: ThemePreference = DEFAULT_THEME_PREFERENCE;

  readonly preference = signal<ThemePreference>(DEFAULT_THEME_PREFERENCE);
  readonly persistenceError = signal<string | null>(null);
  readonly persistenceSaved = signal(false);
  readonly databaseSwitching = signal(false);
  readonly resolvedTheme = computed<ResolvedTheme>(() => {
    const preference = this.preference();
    return preference === 'system' ? (this.systemPrefersDark() ? 'dark' : 'light') : preference;
  });

  constructor() {
    this.persistedPreference = this.readLegacyPreference() ?? DEFAULT_THEME_PREFERENCE;
    this.preference.set(this.persistedPreference);
    this.initializeSystemPreference();
    this.applyResolvedTheme(this.resolvedTheme());
  }

  setPreference(preference: ThemePreference): Promise<void> {
    if (!isThemePreference(preference) || this.databaseSwitching()) return Promise.resolve();
    const revision = ++this.preferenceRevision;
    this.applyPreference(preference);
    this.persistenceSaved.set(false);
    this.persistenceError.set(null);
    const pending = this.persistenceQueue.then(async () => {
      if (revision !== this.preferenceRevision) return;
      const response = await this.settingsApi.updateSettings({ theme_preference: preference });
      if (revision !== this.preferenceRevision) return;
      if (!response.result) {
        this.persistenceError.set(response.error ?? 'Unable to save the theme preference.');
        this.persistenceSaved.set(false);
        this.applyPreference(this.persistedPreference);
        return;
      }
      this.persistedPreference = response.result.values.theme_preference ?? DEFAULT_THEME_PREFERENCE;
      this.storage.removeString(LEGACY_THEME_STORAGE_KEY);
      this.persistenceError.set(null);
      this.persistenceSaved.set(true);
      this.applyPreference(this.persistedPreference);
    });
    this.persistenceQueue = pending.catch(() => undefined);
    return pending.catch((error: unknown) => {
      if (revision !== this.preferenceRevision) return;
      this.persistenceError.set(error instanceof Error ? error.message : 'Unable to save the theme preference.');
      this.persistenceSaved.set(false);
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

    const legacyPreference = this.readLegacyPreference();
    if (legacyPreference && response.result.values.theme_preference === null) {
      const migrated = await this.settingsApi.updateSettings({ theme_preference: legacyPreference });
      if (revision !== this.preferenceRevision) return;
      if (!migrated.result) {
        this.persistenceError.set(migrated.error ?? 'Unable to migrate the saved theme preference.');
        this.persistedPreference = DEFAULT_THEME_PREFERENCE;
        this.applyPreference(legacyPreference);
        return;
      }
      this.storage.removeString(LEGACY_THEME_STORAGE_KEY);
      this.persistenceError.set(null);
      this.applyDatabasePreference(migrated.result.values.theme_preference ?? legacyPreference);
      return;
    }

    this.storage.removeString(LEGACY_THEME_STORAGE_KEY);
    this.persistenceError.set(null);
    this.applyDatabasePreference(response.result.values.theme_preference);
  }

  applyDatabasePreference(preference: ThemePreference | null): void {
    if (preference !== null && !isThemePreference(preference)) return;
    this.preferenceRevision += 1;
    this.persistedPreference = preference ?? DEFAULT_THEME_PREFERENCE;
    this.storage.removeString(LEGACY_THEME_STORAGE_KEY);
    this.persistenceError.set(null);
    this.persistenceSaved.set(false);
    this.applyPreference(this.persistedPreference);
  }

  setDatabaseSwitching(switching: boolean): void {
    this.databaseSwitching.set(switching);
  }

  async flushPersistence(): Promise<void> {
    await this.persistenceQueue;
  }

  clearLegacyPreference(): void {
    this.storage.removeString(LEGACY_THEME_STORAGE_KEY);
  }

  private readLegacyPreference(): ThemePreference | null {
    const stored = this.storage.readString(LEGACY_THEME_STORAGE_KEY);
    return isThemePreference(stored) ? stored : null;
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
