export const THEME_STORAGE_KEY = 'theme-preference';
export const DEFAULT_THEME_PREFERENCE = 'system' as const;

export const THEME_PREFERENCES = ['light', 'dark', 'system'] as const;

export type ThemePreference = typeof THEME_PREFERENCES[number];
export type ResolvedTheme = Exclude<ThemePreference, 'system'>;

export function isThemePreference(value: unknown): value is ThemePreference {
  return typeof value === 'string' && (THEME_PREFERENCES as readonly string[]).includes(value);
}
